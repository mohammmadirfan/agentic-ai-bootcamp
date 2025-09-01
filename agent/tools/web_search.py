import os
import requests
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Web search tool using Serper API"""
    
    def __init__(self):
        """Initialize the web search tool"""
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            logger.warning("SERPER_API_KEY not found in environment variables")
        
        self.description = "Search the web for current information, news, and real-time data"
        self.capabilities = [
            "Current events and news",
            "Real-time information",
            "Recent developments",
            "Live data and statistics",
            "Current weather and conditions"
        ]
        
        self.base_url = "https://google.serper.dev/search"
    
    def search(self, query: str, num_results: int = 5) -> str:
        """
        Perform web search and return formatted results
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            Formatted search results as string
        """
        if not self.api_key:
            return "❌ Web search is not available (API key not configured)"
        
        try:
            # Prepare search request
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": num_results,
                "hl": "en",
                "gl": "us"
            }
            
            # Make API request
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            response.raise_for_status()
            results = response.json()
            
            # Format results
            return self._format_search_results(results, query)
            
        except requests.exceptions.Timeout:
            return "❌ Web search timed out. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"Web search API error: {e}")
            return f"❌ Web search failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected web search error: {e}")
            return f"❌ Unexpected error during web search: {str(e)}"
    
    def _format_search_results(self, results: Dict[str, Any], query: str) -> str:
        """Format search results into a readable response"""
        try:
            formatted_response = f"🔍 **Web Search Results for: '{query}'**\n\n"
            
            # Check if we have organic results
            organic_results = results.get("organic", [])
            
            if not organic_results:
                return "❌ No search results found for your query."
            
            # Add answer box if available
            if "answerBox" in results:
                answer_box = results["answerBox"]
                if "answer" in answer_box:
                    formatted_response += f"**Quick Answer:** {answer_box['answer']}\n\n"
                elif "snippet" in answer_box:
                    formatted_response += f"**Quick Answer:** {answer_box['snippet']}\n\n"
            
            # Add top results
            formatted_response += "**Top Results:**\n\n"
            
            for i, result in enumerate(organic_results[:3], 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description available")
                link = result.get("link", "")
                
                formatted_response += f"**{i}. {title}**\n"
                formatted_response += f"{snippet}\n"
                if link:
                    formatted_response += f"🔗 Source: {link}\n\n"
            
            # Add knowledge graph if available
            if "knowledgeGraph" in results:
                kg = results["knowledgeGraph"]
                if "description" in kg:
                    formatted_response += f"\n**Quick Facts:** {kg['description']}\n"
            
            # Add related searches
            if "relatedSearches" in results and results["relatedSearches"]:
                formatted_response += "\n**Related Searches:**\n"
                for related in results["relatedSearches"][:3]:
                    formatted_response += f"• {related.get('query', '')}\n"
            
            formatted_response += f"\n*Search completed at {datetime.now().strftime('%H:%M:%S')}*"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            return f"❌ Error formatting search results: {str(e)}"
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions for autocomplete (optional feature)"""
        try:
            if not self.api_key or len(partial_query) < 3:
                return []
            
            # This would require a different API endpoint for suggestions
            # For now, return some common search patterns
            suggestions = [
                f"{partial_query} latest news",
                f"{partial_query} 2024",
                f"what is {partial_query}",
                f"{partial_query} trends"
            ]
            
            return suggestions[:3]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return []