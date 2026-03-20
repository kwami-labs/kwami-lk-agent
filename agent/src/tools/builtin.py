"""Built-in function tools for KwamiAgent."""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from livekit.agents import RunContext, function_tool

from ..room_context import get_current_room
from ..utils.logging import get_logger
from ..constants import (
    CartesiaVoices,
    LANGUAGE_GREETINGS,
    TTSProviders,
)

logger = get_logger("tools")


# Price patterns: $12.99, €199, £50, 99 EUR, 1,200€, 49.99 USD, etc.
_PRICE_RE = re.compile(
    r"(?:^|[\s,(])((?:USD|EUR|GBP|€|£|\$)\s*)?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?|\d+[.,]?\d*)\s*(USD|EUR|GBP|€|£|\$)?(?:[\s,]|$)",
    re.IGNORECASE,
)


def _extract_price(text: str) -> Optional[str]:
    """Extract first price-like string from text for product cards (e.g. '€199', '$49.99')."""
    if not (text or "").strip():
        return None
    m = _PRICE_RE.search(text)
    if not m:
        return None
    prefix = (m.group(1) or "").strip()
    num = (m.group(2) or "").replace(",", ".")
    suffix = (m.group(3) or "").strip()
    if not num:
        return None
    currency = prefix or suffix or ""
    if currency.upper() in ("USD", "EUR", "GBP"):
        currency = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency.upper(), currency)
    return f"{currency}{num}".strip() if currency else num.strip()


def _product_name_from_title(title: str) -> str:
    """Short product name: strip site name, ' - Site', ' | Site', etc."""
    if not title:
        return ""
    t = title.strip()
    for sep in (" | ", " – ", " - ", " — "):
        if sep in t:
            t = t.split(sep)[0].strip()
    return t[:80] if t else ""


def _extract_features(content: str, max_items: int = 8) -> List[str]:
    """Extract short feature-like phrases from snippet (apartments, products, events)."""
    if not (content or "").strip():
        return []
    # Split on newlines, bullets, dashes, semicolons, and commas (listings often use commas)
    raw = re.split(r"[\n•\-;,]+|\s+-\s+", content)
    seen: set = set()
    features: List[str] = []
    for part in raw:
        s = (part or "").strip()
        if not s or len(s) < 2:
            continue
        if len(s) > 72:
            s = s[:69] + "..."
        low = s.lower()
        if low in ("and", "or", "the", "with", "for", "from", "in", "to"):
            continue
        if low in seen:
            continue
        seen.add(low)
        features.append(s)
        if len(features) >= max_items:
            break
    return features[:max_items]


async def _tavily_extract_images(
    api_key: str,
    urls: List[str],
    timeout: float = 12.0,
    usage_tracker: Any = None,
) -> Dict[str, List[str]]:
    """Extract content and images from URLs via Tavily Extract. Returns url -> list of image URLs."""
    out: Dict[str, List[str]] = {u: [] for u in urls}
    if not api_key or not urls:
        return out
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                "https://api.tavily.com/extract",
                json={"urls": urls[:5], "include_images": True, "extract_depth": "basic"},
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            r.raise_for_status()
            if usage_tracker:
                usage_tracker.record_external_usage(
                    "tool",
                    "tavily/extract",
                    units_used=1.0,
                    request_count=1,
                )
            data = r.json()
            for item in (data.get("results") or []):
                url = item.get("url")
                images = item.get("images") or []
                if url and isinstance(images, list):
                    out[url] = [img for img in images if isinstance(img, str) and img.startswith("http")][:3]
    except Exception as e:
        logger.debug("Tavily extract failed: %s", e)
    return out


async def _fetch_image_for_url(
    url: str,
    timeout: float = 3.5,
    usage_tracker: Any = None,
) -> Optional[str]:
    """Try to get og:image or primary image for a URL via Microlink (no key required)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(
                "https://api.microlink.io/",
                params={"url": url, "screenshot": "false", "video": "false"},
            )
            r.raise_for_status()
            if usage_tracker:
                usage_tracker.record_external_usage(
                    "tool",
                    "microlink/fetch",
                    units_used=1.0,
                    request_count=1,
                )
            data = r.json()
            d = data.get("data") or {}
            img = d.get("image")
            url_out = None
            if isinstance(img, dict) and img.get("url"):
                url_out = img["url"]
            elif isinstance(img, str) and img.startswith("http"):
                url_out = img
            if not url_out and d.get("logo"):
                logo = d.get("logo")
                if isinstance(logo, dict) and logo.get("url"):
                    url_out = logo["url"]
                elif isinstance(logo, str) and logo.startswith("http"):
                    url_out = logo
            return url_out
    except Exception as e:
        logger.debug("Microlink fetch failed for %s: %s", url[:50], e)
    return None


def _is_elevenlabs_tts(tts: Any) -> bool:
    """Check if TTS provider is ElevenLabs.
    
    Handles both the direct ElevenLabs plugin (livekit.plugins.elevenlabs)
    and LiveKit Inference TTS with an ElevenLabs model (livekit.agents.inference.tts).
    """
    provider = getattr(tts, "provider", "").lower()
    # Check the model string for "elevenlabs" (covers inference.TTS with elevenlabs model)
    model = str(getattr(tts, "_model", getattr(tts, "model", ""))).lower()
    return (
        provider == TTSProviders.ELEVENLABS
        or "elevenlabs" in type(tts).__module__
        or "elevenlabs" in model
    )


class AgentToolsMixin:
    """Mixin containing function tools for KwamiAgent.
    
    This mixin assumes the following attributes exist on the class:
    - kwami_config: KwamiConfig instance
    - _current_voice_config: KwamiVoiceConfig instance
    - _memory: Optional KwamiMemory instance
    - session: AgentSession with tts and stt attributes
    """

    @function_tool()
    async def get_kwami_info(self, context: RunContext) -> Dict[str, Any]:
        """Get information about this Kwami instance."""
        return {
            "kwami_id": self.kwami_config.kwami_id,
            "kwami_name": self.kwami_config.kwami_name,
            "soul": {
                "name": self.kwami_config.soul.name,
                "personality": self.kwami_config.soul.personality,
            },
            # Backward compatibility for older clients still reading "persona".
            "persona": {
                "name": self.kwami_config.soul.name,
                "personality": self.kwami_config.soul.personality,
            },
        }

    @function_tool()
    async def get_current_time(self, context: RunContext) -> str:
        """Get the current time. Useful when the user asks what time it is."""
        from datetime import datetime
        return datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")

    @function_tool()
    async def change_voice(self, context: RunContext, voice_name: str) -> str:
        """Change the TTS voice. Available voices depend on the current TTS provider.
        
        Args:
            voice_name: The name or ID of the voice to switch to.
                       For Cartesia: Use voice names like 'British Lady', 'California Girl', etc.
                       For ElevenLabs: Use voice names like 'Rachel', 'Josh', 'Bella', etc.
                       For OpenAI: Use 'alloy', 'echo', 'nova', 'shimmer', 'onyx', 'fable'.
        """
        try:
            if not hasattr(self, "session") or self.session is None:
                return "Unable to change voice - session not available"
            
            if self.session.tts is None:
                return "Unable to change voice - TTS not available"
            
            # Check if it's a known name and convert to ID
            voice_id = CartesiaVoices.NAME_MAP.get(voice_name.lower(), voice_name)
            
            # Different TTS providers use different parameter names
            if _is_elevenlabs_tts(self.session.tts):
                self.session.tts.update_options(voice_id=voice_id)
            else:
                self.session.tts.update_options(voice=voice_id)
            
            logger.info(f"Voice changed to: {voice_name}")
            return f"Voice changed to {voice_name}. I'm now speaking with a different voice!"
            
        except Exception as e:
            logger.error(f"Failed to change voice: {e}")
            return f"Sorry, I couldn't change the voice: {str(e)}"

    @function_tool()
    async def change_speaking_speed(self, context: RunContext, speed: float) -> str:
        """Change the speaking speed. 
        
        Args:
            speed: Speed multiplier between 0.5 (slow) and 2.0 (fast). 
                   1.0 is normal speed.
        """
        try:
            if not hasattr(self, "session") or self.session is None:
                return "Unable to change speed - session not available"
            
            if self.session.tts is None:
                return "Unable to change speed - TTS not available"
            
            speed = max(0.5, min(2.0, speed))  # Clamp to valid range
            
            # ElevenLabs TTS does not support speed option
            if _is_elevenlabs_tts(self.session.tts):
                return "Speed adjustment is not supported with the current ElevenLabs voice provider."
            
            self.session.tts.update_options(speed=speed)
            logger.info(f"Speaking speed changed to: {speed}")
            
            if speed < 0.8:
                return f"Speed set to {speed}. I'll speak more slowly now."
            elif speed > 1.2:
                return f"Speed set to {speed}. I'll speak faster now."
            else:
                return f"Speed set to {speed}. Speaking at normal pace."
                
        except Exception as e:
            logger.error(f"Failed to change speed: {e}")
            return f"Sorry, I couldn't change the speed: {str(e)}"

    @function_tool()
    async def change_language(self, context: RunContext, language: str) -> str:
        """Change the conversation language for both speech recognition and synthesis.
        
        Args:
            language: Language code like 'en' (English), 'es' (Spanish), 'fr' (French),
                     'de' (German), 'it' (Italian), 'pt' (Portuguese), 'ja' (Japanese),
                     'ko' (Korean), 'zh' (Chinese).
        """
        try:
            if not hasattr(self, "session") or self.session is None:
                return f"Language preference noted: {language}"
            
            language = language.lower().strip()
            
            # Update STT language
            if self.session.stt is not None:
                self.session.stt.update_options(language=language)
                logger.info(f"STT language changed to: {language}")
            
            # Update TTS language if supported
            if self.session.tts is not None:
                try:
                    self.session.tts.update_options(language=language)
                    logger.info(f"TTS language changed to: {language}")
                except Exception:
                    pass  # Not all TTS providers support language parameter
            
            return LANGUAGE_GREETINGS.get(language, f"Language changed to {language}.")
            
        except Exception as e:
            logger.error(f"Failed to change language: {e}")
            return f"Sorry, I couldn't change the language: {str(e)}"

    @function_tool()
    async def get_current_voice_settings(self, context: RunContext) -> Dict[str, Any]:
        """Get the current voice pipeline settings."""
        voice_config = self._current_voice_config
        return {
            "tts_provider": voice_config.tts_provider,
            "tts_model": voice_config.tts_model,
            "tts_voice": voice_config.tts_voice,
            "tts_speed": voice_config.tts_speed,
            "stt_provider": voice_config.stt_provider,
            "stt_model": voice_config.stt_model,
            "stt_language": voice_config.stt_language,
            "llm_provider": voice_config.llm_provider,
            "llm_model": voice_config.llm_model,
            "llm_temperature": voice_config.llm_temperature,
        }

    @function_tool()
    async def remember_fact(self, context: RunContext, fact: str) -> str:
        """Remember an important fact about the user for future conversations."""
        if not self._memory or not self._memory.is_initialized:
            return "Memory is not available in this session."
        
        try:
            await self._memory.add_fact(fact)
            logger.info(f"Remembered fact: {fact}")
            return f"I'll remember that: {fact}"
        except Exception as e:
            logger.error(f"Failed to remember fact: {e}")
            return "Sorry, I couldn't save that to memory."

    @function_tool()
    async def recall_memories(self, context: RunContext, topic: str) -> str:
        """Search your memory for information about a specific topic."""
        if not self._memory or not self._memory.is_initialized:
            return "Memory is not available in this session."
        
        try:
            results = await self._memory.search(topic, limit=5)
            
            if not results:
                return f"I don't have any memories about '{topic}' yet."
            
            memories = []
            for r in results:
                if r.get("content"):
                    memories.append(f"- {r['content']}")
            
            if memories:
                return f"Here's what I remember about '{topic}':\n" + "\n".join(memories)
            return f"I don't have specific memories about '{topic}'."
            
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return "Sorry, I couldn't search my memory right now."

    @function_tool()
    async def get_memory_status(self, context: RunContext) -> Dict[str, Any]:
        """Get the current memory status and statistics."""
        if not self._memory:
            return {
                "enabled": False,
                "status": "Memory not configured",
            }
        
        if not self._memory.is_initialized:
            return {
                "enabled": True,
                "status": "Memory not initialized",
            }
        
        try:
            memory_context = await self._memory.get_context()
            return {
                "enabled": True,
                "status": "Active",
                "user_id": self._memory.user_id,
                "session_id": self._memory.session_id,
                "facts_count": len(memory_context.facts),
                "recent_messages_count": len(memory_context.recent_messages),
                "has_summary": memory_context.summary is not None,
            }
        except Exception as e:
            return {
                "enabled": True,
                "status": f"Error: {str(e)}",
            }

    @function_tool()
    async def product_search(self, context: RunContext, query: str, max_results: int = 5) -> str:
        """Search for actual products to buy (bags, clothes, gifts, etc.). Returns real product cards with image, name, and price—not store homepages.
        Use this when the user wants to find or buy something (e.g. 'bags for my girlfriend', 'women's jackets', 'gift ideas').
        Prefer this over web_search for any shopping or product-finding intent.

        Args:
            query: Product search query (e.g. "women's bags", "leather handbags", "gift for girlfriend").
            max_results: Number of products to return (1-10, default 5).
        """
        api_key = os.environ.get("SERPAPI_KEY")
        if not api_key:
            logger.info("SERPAPI_KEY not set; product_search unavailable, use web_search with search_for_products")
            return (
                "Product search is not configured (set SERPAPI_KEY for real product results). "
                "Use web_search with search_for_products=True instead."
            )
        max_results = max(1, min(10, max_results))
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                r = await client.get(
                    "https://serpapi.com/search",
                    params={"engine": "google_shopping", "q": query, "api_key": api_key, "num": max_results},
                )
                r.raise_for_status()
                if getattr(self, "usage_tracker", None):
                    self.usage_tracker.record_external_usage(
                        "tool",
                        "serpapi/google_shopping",
                        units_used=1.0,
                        request_count=1,
                    )
                data = r.json()
        except Exception as e:
            logger.warning("SerpApi product search failed: %s", e)
            return "Product search failed. Try web_search with search_for_products=True."
        shopping = data.get("shopping_results") or []
        if not shopping:
            return "No products found. Try web_search with a different query."
        # Build product cards: real product image, title, price, link
        ui_results: List[Dict[str, Any]] = []
        for p in shopping[:max_results]:
            title = (p.get("title") or "")[:180]
            price_str = p.get("price") or ""
            link = p.get("product_link") or p.get("link") or ""
            thumb = p.get("thumbnail") or ""
            snippet = (p.get("snippet") or "")[:200]
            source = p.get("source") or ""
            features = [f for f in [source, snippet] if f][:3]
            ui_results.append({
                "title": title,
                "product_name": title[:120],
                "price": price_str[:30] if price_str else None,
                "url": link[:500] if link else "",
                "content": snippet,
                "image": (thumb[:500] if thumb else None),
                "features": features,
            })
        answer = f"Found {len(ui_results)} products for '{query[:50]}'."
        if self._memory and self._memory.is_initialized:
            try:
                await self._memory.add_fact(f"User searched for products: {query}")
            except Exception:
                pass
        room = get_current_room() or (getattr(context, "room", None) if context else None) or self.room
        if room:
            try:
                msg = {
                    "type": "search_results",
                    "query": query,
                    "results": ui_results,
                    "answer": answer[:400],
                }
                payload = json.dumps(msg).encode("utf-8")
                if len(payload) > 14 * 1024:
                    for item in msg.get("results", []):
                        item["content"] = (item.get("content") or "")[:80]
                payload = json.dumps(msg).encode("utf-8")
                await room.local_participant.publish_data(payload, reliable=True)
                logger.info("Published product_search results: %s products", len(ui_results))
            except Exception as e:
                logger.warning("Failed to send product results to client: %s", e)
        return answer

    @function_tool()
    async def web_search(
        self,
        context: RunContext,
        query: str,
        max_results: int = 5,
        search_for_products: bool = False,
    ) -> str:
        """Search the web for current information. Use for facts, news, or when product_search is not available.

        For shopping or gifts: prefer product_search so the user sees real product cards (image, name, price). Use web_search with search_for_products=True only if product_search is not configured.

        Args:
            query: The search query.
            max_results: Maximum number of results to return (1-10, default 5).
            search_for_products: Set True when the user is looking for products and product_search is unavailable.
        """
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            logger.warning("TAVILY_API_KEY not set; web search disabled")
            return "Web search is not configured (missing TAVILY_API_KEY)."

        max_results = max(1, min(10, max_results))
        payload = {
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": True,
        }
        if search_for_products:
            payload["include_domains"] = [
                "amazon.com", "etsy.com", "asos.com", "nordstrom.com", "zara.com",
                "hm.com", "macy.com", "bloomingdales.com", "ssense.com", "farfetch.com",
            ]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                if getattr(self, "usage_tracker", None):
                    self.usage_tracker.record_external_usage(
                        "tool",
                        "tavily/search",
                        units_used=1.0,
                        request_count=1,
                    )
                data = resp.json()
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text
            except Exception:
                pass
            logger.warning(
                "Tavily search failed: status=%s body=%s",
                e.response.status_code,
                body[:500] if body else "",
            )
            if e.response.status_code == 432:
                # Don't echo Tavily's body to the user; it can be wrong (e.g. "usage limit" when credits are fine).
                # We only log it for debugging.
                return "Web search is temporarily unavailable. Please try again in a moment."
            if e.response.status_code == 401:
                return "Web search is not configured correctly (invalid API key)."
            if e.response.status_code == 429:
                return "Web search rate limit reached. Please try again in a moment."
            try:
                err = e.response.json()
                msg = err.get("detail") or err.get("message") or err.get("error") or body
            except Exception:
                msg = body or str(e)
            return f"Search failed: {msg}"
        except Exception as e:
            logger.exception("Tavily search failed")
            return f"Search failed: {str(e)}"

        results: List[Dict[str, Any]] = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in data.get("results", [])
        ]
        answer = data.get("answer") or ""

        # Store search query in memory for future context
        if self._memory and self._memory.is_initialized:
            try:
                await self._memory.add_fact(f"User searched for: {query}")
                if search_for_products:
                    await self._memory.add_fact(f"User is shopping / looking for products: {query}")
            except Exception as e:
                logger.debug("Could not store search in memory: %s", e)

        # Enrich results: extract features, fetch image per result (for UI)
        max_results = 5
        max_content = 220  # Keep payload smaller so images fit under LiveKit limit
        ui_results: List[Dict[str, Any]] = []
        for r in results[:max_results]:
            raw_title = (r.get("title") or "")[:180]
            content = (r.get("content") or "")[:max_content]
            features = _extract_features(content)[:5]  # Max 5 features, shorter list
            price = _extract_price(content) or _extract_price(raw_title)
            product_name = _product_name_from_title(raw_title) or raw_title
            ui_results.append({
                "title": raw_title,
                "product_name": product_name[:120],
                "price": price[:30] if price else None,
                "url": (r.get("url") or "")[:400],
                "content": content,
                "features": [f[:60] for f in features],
            })

        # Get images: prefer Tavily Extract (real page images), fallback to Microlink
        result_urls = [u["url"] for u in ui_results]
        extract_images = await _tavily_extract_images(
            api_key,
            result_urls,
            usage_tracker=getattr(self, "usage_tracker", None),
        )
        for i, u in enumerate(ui_results):
            imgs = extract_images.get(u["url"]) or []
            if imgs:
                ui_results[i]["image"] = imgs[0][:400]

        async def add_image_fallback(idx: int, url: str) -> None:
            if ui_results[idx].get("image"):
                return
            img = await _fetch_image_for_url(
                url,
                usage_tracker=getattr(self, "usage_tracker", None),
            )
            if img and idx < len(ui_results):
                ui_results[idx]["image"] = (img or "")[:400]

        await asyncio.gather(*[add_image_fallback(i, r["url"]) for i, r in enumerate(ui_results)])
        images_count = sum(1 for u in ui_results if u.get("image"))
        logger.info("Fetched %s images for %s results", images_count, len(ui_results))

        room = get_current_room() or (getattr(context, "room", None) if context else None) or self.room
        if room:
            try:
                max_answer = 400
                ui_answer = (answer or "")[:max_answer]
                msg = {
                    "type": "search_results",
                    "query": query,
                    "results": ui_results,
                    "answer": ui_answer,
                }
                payload = json.dumps(msg).encode("utf-8")
                # Prefer keeping images: trim content/answer/features first, strip images only as last resort
                if len(payload) > 14 * 1024:
                    for item in msg.get("results", []):
                        item["content"] = (item.get("content") or "")[:120]
                        item["features"] = (item.get("features") or [])[:3]
                    msg["answer"] = (ui_answer or "")[:150]
                    payload = json.dumps(msg).encode("utf-8")
                if len(payload) > 14 * 1024:
                    for item in msg.get("results", []):
                        item.pop("image", None)
                    payload = json.dumps(msg).encode("utf-8")
                logger.info(
                    "Publishing search_results to room (query=%r, results=%s, with_images=%s)",
                    query[:80] if query else "",
                    len(ui_results),
                    any(u.get("image") for u in msg.get("results", [])),
                )
                await room.local_participant.publish_data(
                    payload,
                    reliable=True,
                )
                logger.info("Published search_results to client")
            except Exception as e:
                logger.warning("Failed to send search_results to client: %s", e)
        else:
            logger.warning(
                "Cannot send search_results to client: no room (self.room=%s, context.room=%s)",
                self.room is not None,
                getattr(context, "room", None) is not None if context else False,
            )

        if answer:
            return answer
        if results:
            return "\n".join(
                f"- {r['title']}: {r['content'][:150]}..." for r in results[:3]
            )
        return "No results found."

    # -------------------------------------------------------------------------
    # Navigation tools (client-side: agent sends commands, client renders)
    # -------------------------------------------------------------------------

    async def _send_nav_command(self, context: RunContext, action: str, **kwargs) -> bool:
        """Send a navigation command to the client via data channel."""
        room = (
            get_current_room()
            or (getattr(context, "room", None) if context else None)
            or self.room
        )
        if not room:
            return False
        msg: Dict[str, Any] = {"type": "nav_command", "action": action, **kwargs}
        await room.local_participant.publish_data(
            json.dumps(msg).encode("utf-8"), reliable=True
        )
        return True

    @function_tool()
    async def navigate_to(self, context: RunContext, url: str) -> str:
        """Open a website for the user. The page opens directly in their browser so they can interact with it.
        Use this when the user asks you to open, go to, visit, or browse a website.

        Args:
            url: The URL to navigate to (e.g. 'https://wikipedia.org', 'google.com').
        """
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        self._last_nav_page_content = ""
        ok = await self._send_nav_command(context, "navigate", url=url)
        if not ok:
            return "Cannot open browser: no room connection available."
        logger.info("Sent navigate command to client: %s", url[:80])
        return f"Opening {url} for the user. They can interact with the page directly."

    @function_tool()
    async def go_back_in_browser(self, context: RunContext) -> str:
        """Go back to the previous page in the user's navigation window."""
        ok = await self._send_nav_command(context, "back")
        if not ok:
            return "Cannot send command: no room connection."
        return "Going back to the previous page."

    @function_tool()
    async def go_forward_in_browser(self, context: RunContext) -> str:
        """Go forward to the next page in the user's navigation window."""
        ok = await self._send_nav_command(context, "forward")
        if not ok:
            return "Cannot send command: no room connection."
        return "Going forward to the next page."

    @function_tool()
    async def close_navigation(self, context: RunContext) -> str:
        """Close the user's navigation window. Use when the user is done browsing or asks to close/stop navigation."""
        self._last_nav_page_content = ""
        ok = await self._send_nav_command(context, "close")
        if not ok:
            return "Cannot send command: no room connection."
        return "Browser closed."

    @function_tool()
    async def click_in_navigation(
        self,
        context: RunContext,
        element_description: str = "",
        element_id: str = "",
    ) -> str:
        """Click an element on the page currently open in the navigation window.
        Use the page content (from read_navigation_page) to see the HTML and the list of interactive elements with ids (el-0, el-1, ...). Prefer element_id when you can identify the exact element from the list (e.g. element_id='el-5' for the 6th element). Otherwise use element_description (e.g. 'search button', 'first video', 'Sign in').

        Args:
            element_description: Description of the element to click (e.g. 'search button', 'first video'). Use when element_id is not set.
            element_id: Exact element id from the page content list (e.g. 'el-5'). Use when you have read the page and know which id to click.
        """
        kwargs = {}
        if element_id and element_id.strip().startswith("el-"):
            kwargs["element_id"] = element_id.strip()
        if element_description and element_description.strip():
            kwargs["description"] = element_description.strip()
        if not kwargs:
            return "Specify either element_description or element_id."
        ok = await self._send_nav_command(context, "click", **kwargs)
        if not ok:
            return "Cannot send command: no room connection."
        if kwargs.get("element_id"):
            return f"Clicking element {kwargs['element_id']} for the user."
        return f"Clicking on '{kwargs.get('description', '')}' for the user."

    @function_tool()
    async def type_in_navigation(
        self,
        context: RunContext,
        text: str,
        field_description: str = "",
        element_id: str = "",
        clear_first: bool = True,
    ) -> str:
        """Type text into a field on the page currently open in the navigation window.

        Args:
            text: The text to type.
            field_description: Optional description of which field to type into (e.g. 'search box'). If empty, types into the currently focused field.
            element_id: Exact element id from read_navigation_page (e.g. 'el-3'). Preferred when you know the id.
            clear_first: If True (default), clears the field before typing. Set False to append.
        """
        desc = field_description
        if clear_first:
            desc = f"__clear__{desc}"
        kwargs: Dict[str, Any] = {"inputText": text, "description": desc}
        if element_id and element_id.strip().startswith("el-"):
            kwargs["element_id"] = element_id.strip()
        ok = await self._send_nav_command(context, "type", **kwargs)
        if not ok:
            return "Cannot send command: no room connection."
        parts = [f"Typing '{text}'"]
        if element_id:
            parts.append(f"in element {element_id}")
        elif field_description:
            parts.append(f"in '{field_description}'")
        return " ".join(parts)

    @function_tool()
    async def press_key_in_navigation(self, context: RunContext, key: str) -> str:
        """Press a keyboard key on the navigation page (e.g. Enter to submit a search).

        Args:
            key: Key to press (e.g. 'Enter', 'Tab', 'Escape').
        """
        ok = await self._send_nav_command(context, "press_key", description=key)
        if not ok:
            return "Cannot send command: no room connection."
        return f"Pressed '{key}'."

    @function_tool()
    async def scroll_navigation(self, context: RunContext, direction: str = "down") -> str:
        """Scroll the navigation page up or down.

        Args:
            direction: 'up' or 'down'.
        """
        ok = await self._send_nav_command(context, "scroll", description=direction)
        if not ok:
            return "Cannot send command: no room connection."
        return f"Scrolled {direction}."

    @function_tool()
    async def read_navigation_page(self, context: RunContext) -> str:
        """Read the content of the page currently open in the navigation window.
        Returns the page title, main text, a list of interactive elements with ids (el-0, el-1, ...), and the HTML structure so you see what the user sees. Use this to decide what to click; then use click_in_navigation with element_id='el-N' for precise clicks."""
        prev = getattr(self, "_last_nav_page_content", "") or ""
        self._last_nav_page_content = ""
        ok = await self._send_nav_command(context, "read_page")
        if not ok:
            return "Cannot send command: no room connection."
        for _ in range(10):
            await asyncio.sleep(0.3)
            content = getattr(self, "_last_nav_page_content", "") or ""
            if content:
                return content
        if prev:
            self._last_nav_page_content = prev
            return prev
        return "Page content not available yet. The page may still be loading — try again in a moment."

    # -------------------------------------------------------------------------
    # Search result management
    # -------------------------------------------------------------------------

    @function_tool()
    async def dismiss_search_result(self, context: RunContext, index: int) -> str:
        """Remove a search result card from the user's screen. Call this when the user says to discard, remove, or dismiss a result (e.g. 'discard the first one', 'remove that', 'not that one').
        Use the 0-based index: first card is 0, second is 1, and so on.
        Args:
            index: 0-based index of the result to remove (0 = first card, 1 = second, etc.).
        """
        room = get_current_room() or (getattr(context, "room", None) if context else None) or self.room
        if room:
            try:
                msg = {"type": "remove_result", "index": max(0, int(index))}
                await room.local_participant.publish_data(
                    json.dumps(msg).encode("utf-8"),
                    reliable=True,
                )
                return f"Removed that result from the screen."
            except Exception as e:
                logger.warning("Failed to send remove_result to client: %s", e)
        return "Could not remove the result."
