# news_search.py
# ==============================================================================
# Real-Time External Intelligence: News Search + Weather API
# Uses DuckDuckGo (free) + Open-Meteo (free, no API key)
# LLM-powered analysis via Groq for smart risk assessment
# ==============================================================================

import json
import time
import urllib.request
from typing import List, Dict, Optional
from datetime import datetime

# DuckDuckGo for news (prefer new 'ddgs' package)
try:
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDG = True
    except ImportError:
        HAS_DDG = False


# ==============================================================================
# WEATHER SERVICE — Open-Meteo API (Free, No Key)
# ==============================================================================

class WeatherService:
    """Fetch real-time weather for our store location (California).
    
    Uses Open-Meteo API — completely free, no API key needed.
    Location: Los Angeles area (CA_1 Walmart store).
    """

    # Los Angeles, California (approximate CA_1 Walmart location)
    LATITUDE = 34.05
    LONGITUDE = -118.24
    LOCATION_NAME = "Los Angeles, California"

    # WMO Weather interpretation codes
    WMO_CODES = {
        0: ("☀️", "Clear sky"),
        1: ("🌤️", "Mainly clear"), 2: ("⛅", "Partly cloudy"), 3: ("☁️", "Overcast"),
        45: ("🌫️", "Foggy"), 48: ("🌫️", "Icy fog"),
        51: ("🌦️", "Light drizzle"), 53: ("🌦️", "Moderate drizzle"), 55: ("🌧️", "Dense drizzle"),
        61: ("🌧️", "Slight rain"), 63: ("🌧️", "Moderate rain"), 65: ("🌧️", "Heavy rain"),
        66: ("🌨️", "Freezing rain"), 67: ("🌨️", "Heavy freezing rain"),
        71: ("🌨️", "Slight snow"), 73: ("❄️", "Moderate snow"), 75: ("❄️", "Heavy snow"),
        77: ("❄️", "Snow grains"),
        80: ("🌧️", "Rain showers"), 81: ("🌧️", "Moderate showers"), 82: ("⛈️", "Violent showers"),
        85: ("🌨️", "Snow showers"), 86: ("❄️", "Heavy snow showers"),
        95: ("⛈️", "Thunderstorm"), 96: ("⛈️", "Thunderstorm + hail"), 99: ("⛈️", "Severe thunderstorm"),
    }

    def __init__(self):
        self.last_weather = None
        self.last_fetch_time = 0
        self.cache_seconds = 300  # Cache for 5 minutes

    def fetch_weather(self) -> Optional[Dict]:
        """Fetch current weather from Open-Meteo API."""
        # Rate limit: don't fetch more than once per 5 minutes
        if self.last_weather and (time.time() - self.last_fetch_time) < self.cache_seconds:
            return self.last_weather

        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={self.LATITUDE}&longitude={self.LONGITUDE}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"precipitation,rain,wind_speed_10m,wind_gusts_10m,weather_code"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"wind_speed_10m_max,weather_code"
            f"&timezone=America%2FLos_Angeles&forecast_days=3"
        )

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SupplyChainAI/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            current = data.get('current', {})
            daily = data.get('daily', {})
            weather_code = current.get('weather_code', 0)
            emoji, description = self.WMO_CODES.get(weather_code, ("🌡️", "Unknown"))

            # Check for severe weather
            wind = current.get('wind_speed_10m', 0)
            gusts = current.get('wind_gusts_10m', 0)
            rain = current.get('precipitation', 0)
            temp = current.get('temperature_2m', 20)

            severity = 'normal'
            if weather_code >= 95 or gusts > 80 or rain > 20:
                severity = 'severe'
            elif weather_code >= 61 or gusts > 50 or rain > 5:
                severity = 'moderate'

            # 3-day forecast summary
            forecast_3day = []
            if daily:
                dates = daily.get('time', [])
                max_temps = daily.get('temperature_2m_max', [])
                min_temps = daily.get('temperature_2m_min', [])
                precip_sums = daily.get('precipitation_sum', [])
                daily_codes = daily.get('weather_code', [])

                for i in range(min(3, len(dates))):
                    fc_code = daily_codes[i] if i < len(daily_codes) else 0
                    fc_emoji, fc_desc = self.WMO_CODES.get(fc_code, ("🌡️", "Unknown"))
                    forecast_3day.append({
                        'date': dates[i] if i < len(dates) else '?',
                        'emoji': fc_emoji,
                        'description': fc_desc,
                        'temp_max': max_temps[i] if i < len(max_temps) else 0,
                        'temp_min': min_temps[i] if i < len(min_temps) else 0,
                        'rain': precip_sums[i] if i < len(precip_sums) else 0,
                    })

            self.last_weather = {
                'location': self.LOCATION_NAME,
                'emoji': emoji,
                'description': description,
                'temperature': temp,
                'feels_like': current.get('apparent_temperature', temp),
                'humidity': current.get('relative_humidity_2m', 0),
                'wind_speed': wind,
                'wind_gusts': gusts,
                'rain': rain,
                'weather_code': weather_code,
                'severity': severity,
                'forecast_3day': forecast_3day,
                'fetched_at': datetime.now().isoformat(),
            }
            self.last_fetch_time = time.time()
            return self.last_weather

        except Exception as e:
            print(f"Weather fetch failed: {e}")
            return self.last_weather  # Return cached if available

    def get_supply_chain_impact(self) -> Dict:
        """Assess weather impact on supply chain operations."""
        w = self.last_weather
        if not w:
            return {'impact': 'none', 'logistics_delay': 0, 'demand_multiplier': 1.0, 'details': 'No weather data'}

        severity = w.get('severity', 'normal')
        wind = w.get('wind_speed', 0)
        rain = w.get('rain', 0)
        temp = w.get('temperature', 20)

        if severity == 'severe':
            return {
                'impact': 'high',
                'logistics_delay': 2,
                'demand_multiplier': 1.5,
                'reliability_adjustment': -0.20,
                'details': f"Severe weather: {w['description']}. Wind {wind} km/h, Rain {rain}mm. "
                          f"Expect delivery delays +2 days, demand spike +50%."
            }
        elif severity == 'moderate':
            return {
                'impact': 'medium',
                'logistics_delay': 1,
                'demand_multiplier': 1.2,
                'reliability_adjustment': -0.10,
                'details': f"Moderate weather: {w['description']}. Wind {wind} km/h, Rain {rain}mm. "
                          f"Possible delays +1 day, slight demand increase."
            }
        else:
            # Extreme heat can also spike grocery demand
            if temp > 38:
                return {
                    'impact': 'medium',
                    'logistics_delay': 0,
                    'demand_multiplier': 1.3,
                    'reliability_adjustment': 0.0,
                    'details': f"Extreme heat: {temp}°C. Expect increased demand for water and perishables (+30%)."
                }
            return {
                'impact': 'low',
                'logistics_delay': 0,
                'demand_multiplier': 1.0,
                'reliability_adjustment': 0.0,
                'details': f"Clear conditions: {w['description']}, {temp}°C. No supply chain impact expected."
            }


# ==============================================================================
# NEWS SEARCH — DuckDuckGo
# ==============================================================================

class SupplyChainNewsSearch:
    """Searches real-time news for supply chain disruptions.
    
    Uses DuckDuckGo as primary source, with Google News RSS as fallback.
    """

    def __init__(self):
        self.last_results = []
        self.available = HAS_DDG

    def _fetch_google_news_rss(self, query: str, max_results: int = 5) -> List[Dict]:
        """Fallback: fetch news from Google News RSS feed (free, no rate limits)."""
        import xml.etree.ElementTree as ET
        encoded_query = urllib.request.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        results = []
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SupplyChainAI/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read().decode('utf-8')
            root = ET.fromstring(xml_data)
            for item in root.findall('.//item')[:max_results]:
                title = item.findtext('title', '')
                link = item.findtext('link', '')
                pub_date = item.findtext('pubDate', '')
                # Google News wraps source in title like "Headline - Source"
                source = ''
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    title = parts[0]
                    source = parts[1] if len(parts) > 1 else ''
                results.append({
                    'title': title,
                    'snippet': '',
                    'url': link,
                    'date': pub_date,
                    'source': source,
                })
        except Exception as e:
            print(f"Google News RSS fallback failed: {e}")
        return results

    def search_disruptions(self, max_results: int = 5) -> List[Dict]:
        """Search for supply chain news. Uses DuckDuckGo with Google News RSS fallback."""
        queries = [
            "supply chain disruptions today",
            "California grocery supply shortage",
        ]

        all_results = []

        # Try DuckDuckGo first
        if self.available:
            try:
                ddgs = DDGS()
                for query in queries[:2]:
                    try:
                        results = list(ddgs.news(query, max_results=3))
                        for r in results:
                            all_results.append({
                                'title': r.get('title', ''),
                                'snippet': r.get('body', r.get('snippet', '')),
                                'url': r.get('url', r.get('link', '')),
                                'date': r.get('date', ''),
                                'source': r.get('source', ''),
                            })
                    except Exception:
                        continue
            except Exception as e:
                print(f"DuckDuckGo search failed (will use fallback): {e}")
                all_results = []

        # Fallback to Google News RSS if DuckDuckGo returned nothing
        if not all_results:
            print("Using Google News RSS fallback...")
            for query in queries[:2]:
                rss_results = self._fetch_google_news_rss(query, max_results=3)
                all_results.extend(rss_results)

        # Deduplicate by title
        seen = set()
        unique = []
        for r in all_results:
            title = r['title']
            if title and title not in seen:
                seen.add(title)
                unique.append(r)

        self.last_results = unique[:max_results]
        return self.last_results


# ==============================================================================
# LLM-POWERED RISK ANALYZER — Uses Groq to UNDERSTAND news + weather
# ==============================================================================

class IntelligenceAnalyzer:
    """Combines weather + news and uses LLM to produce smart risk analysis.
    
    Instead of dumb keyword matching, the LLM reads the actual content
    and produces structured supply chain recommendations.
    """

    def __init__(self):
        self.weather_service = WeatherService()
        self.news_search = SupplyChainNewsSearch()
        self.last_analysis = None
        self.last_analysis_day = -1

    def gather_intelligence(self, simulation_day: int = 0, force: bool = False) -> Dict:
        """Fetch weather + news and return combined raw data."""
        # Fetch weather (cached, cheap)
        weather = self.weather_service.fetch_weather()

        # Fetch news (only every 5 simulation days or when forced)
        news = self.news_search.last_results
        if force or simulation_day == 0 or simulation_day % 5 == 1:
            news = self.news_search.search_disruptions()

        return {
            'weather': weather,
            'news': news,
            'fetched_at': datetime.now().isoformat(),
        }

    def analyze_with_llm(self, llm_engine, simulation_day: int = 0) -> Dict:
        """Use the LLM to analyze combined weather + news intelligence.
        
        Returns structured risk assessment with supply chain recommendations.
        """
        # Don't re-analyze the same day
        if simulation_day == self.last_analysis_day and self.last_analysis:
            return self.last_analysis

        raw = self.gather_intelligence(simulation_day)
        weather = raw.get('weather')
        news = raw.get('news', [])

        # Build context for LLM
        weather_text = "No weather data available."
        if weather:
            weather_text = (
                f"Current weather in {weather['location']}: "
                f"{weather['emoji']} {weather['description']}, "
                f"Temperature: {weather['temperature']}°C (feels like {weather['feels_like']}°C), "
                f"Wind: {weather['wind_speed']} km/h (gusts: {weather['wind_gusts']} km/h), "
                f"Precipitation: {weather['rain']}mm, Humidity: {weather['humidity']}%"
            )
            if weather.get('forecast_3day'):
                forecast_lines = []
                for fc in weather['forecast_3day']:
                    forecast_lines.append(
                        f"  {fc['date']}: {fc['emoji']} {fc['description']}, "
                        f"{fc['temp_min']}-{fc['temp_max']}°C, Rain: {fc['rain']}mm"
                    )
                weather_text += "\n3-Day Forecast:\n" + "\n".join(forecast_lines)

        news_text = "No recent supply chain news."
        if news:
            headlines = [f"  - {n['title']}" for n in news[:5]]
            news_text = "Latest supply chain news headlines:\n" + "\n".join(headlines)

        prompt = f"""You are a supply chain risk analyst for a California Walmart grocery store 
that sells ~66 units/day of a high-demand food product.

CURRENT INTELLIGENCE:
{weather_text}

{news_text}

Analyze the combined impact on our supply chain. Respond in EXACTLY this JSON format:
{{
    "risk_level": "low" or "medium" or "high",
    "risk_score": 0.0 to 1.0,
    "weather_impact": "brief description of weather impact on logistics/demand",
    "news_impact": "brief description of relevant news impact",
    "demand_adjustment": 1.0 to 1.5 (multiplier, 1.0 = no change),
    "logistics_delay_days": 0 to 3,
    "reliability_adjustment": -0.25 to 0.0,
    "recommendation": "one-line recommendation for the supply chain manager",
    "alert_message": "one dramatic alert message for the agent chat (like a news anchor)"
}}

Be realistic. Clear skies = low risk. Only flag disruptions that would actually impact a grocery supply chain."""

        try:
            if llm_engine and hasattr(llm_engine, 'reason'):
                response = llm_engine.reason(
                    "You are a supply chain risk analyst. Respond only in valid JSON.",
                    prompt
                )
                if response:
                    # Parse JSON from response
                    analysis = self._parse_llm_response(response)
                    if analysis:
                        analysis['weather_raw'] = weather
                        analysis['news_raw'] = news
                        analysis['source'] = 'llm'
                        self.last_analysis = analysis
                        self.last_analysis_day = simulation_day
                        return analysis
        except Exception as e:
            print(f"LLM analysis failed: {e}")

        # Fallback: rule-based analysis (if LLM fails)
        return self._fallback_analysis(weather, news, simulation_day)

    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block in response
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except Exception:
            pass
        
        return None

    def _fallback_analysis(self, weather: Optional[Dict], news: List[Dict],
                           simulation_day: int) -> Dict:
        """Fallback rule-based analysis when LLM is unavailable."""
        weather_impact = self.weather_service.get_supply_chain_impact() if weather else {}
        
        risk_level = weather_impact.get('impact', 'low')
        
        analysis = {
            'risk_level': risk_level,
            'risk_score': {'low': 0.1, 'medium': 0.4, 'high': 0.8}.get(risk_level, 0.1),
            'weather_impact': weather_impact.get('details', 'No significant weather impact'),
            'news_impact': f"Found {len(news)} news articles" if news else "No news data",
            'demand_adjustment': weather_impact.get('demand_multiplier', 1.0),
            'logistics_delay_days': weather_impact.get('logistics_delay', 0),
            'reliability_adjustment': weather_impact.get('reliability_adjustment', 0.0),
            'recommendation': 'Continue normal operations' if risk_level == 'low' else 'Monitor situation closely',
            'alert_message': weather_impact.get('details', 'No alerts'),
            'weather_raw': weather,
            'news_raw': news,
            'source': 'rules',
        }
        self.last_analysis = analysis
        self.last_analysis_day = simulation_day
        return analysis

    def get_context_for_agent_prompt(self) -> str:
        """Format the latest analysis for injection into agent LLM prompts."""
        if not self.last_analysis:
            return "No external intelligence available."

        a = self.last_analysis
        lines = [
            f"📡 EXTERNAL INTELLIGENCE (Risk: {a.get('risk_level', '?').upper()}, Score: {a.get('risk_score', 0):.0%})",
            f"🌤️ Weather: {a.get('weather_impact', 'N/A')}",
            f"📰 News: {a.get('news_impact', 'N/A')}",
            f"💡 Recommendation: {a.get('recommendation', 'N/A')}",
        ]
        if a.get('demand_adjustment', 1.0) != 1.0:
            lines.append(f"📈 Demand adjustment: ×{a['demand_adjustment']:.1f}")
        if a.get('logistics_delay_days', 0) > 0:
            lines.append(f"🚚 Logistics delay: +{a['logistics_delay_days']} days")

        return "\n".join(lines)


# ==============================================================================
# SINGLETON ACCESS
# ==============================================================================

_intel_instance = None

def get_intelligence_analyzer() -> IntelligenceAnalyzer:
    global _intel_instance
    if _intel_instance is None:
        _intel_instance = IntelligenceAnalyzer()
    return _intel_instance

# Backward compat
def get_news_searcher():
    return get_intelligence_analyzer().news_search
