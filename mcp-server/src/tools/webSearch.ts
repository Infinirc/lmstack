/**
 * Web Search Tool
 *
 * Uses SearchXNG (open source meta search engine) for web searches.
 * No API key required - uses public instances.
 */

interface SearchResult {
  title: string;
  url: string;
  content?: string;
  snippet?: string;
  engine?: string;
}

interface SearchXNGResponse {
  results?: SearchResult[];
  suggestions?: string[];
  query?: string;
  number_of_results?: number;
}

// Public SearchXNG instances to try
const SEARXNG_INSTANCES = [
  "https://search.infinirc.com",
  "https://searx.be",
  "https://search.sapti.me",
];

/**
 * Search the web using SearchXNG
 *
 * @param query - The search query
 * @param maxResults - Maximum number of results to return (default: 5)
 * @returns Formatted search results as markdown
 */
export async function webSearch(
  query: string,
  maxResults: number = 5
): Promise<string> {
  let lastError: Error | null = null;

  // Try each instance until one works
  for (const instance of SEARXNG_INSTANCES) {
    try {
      const searchUrl = new URL(`${instance}/search`);
      searchUrl.searchParams.set("q", query);
      searchUrl.searchParams.set("format", "json");
      searchUrl.searchParams.set("language", "en");
      searchUrl.searchParams.set("categories", "general");

      const response = await fetch(searchUrl.toString(), {
        headers: {
          Accept: "application/json",
          "User-Agent": "LMStack-Agent/1.0 (https://github.com/lmstack/lmstack)",
        },
        signal: AbortSignal.timeout(15000), // 15 second timeout
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: SearchXNGResponse = await response.json();

      // Format results
      const results = data.results?.slice(0, maxResults) || [];
      let formatted = `## Web Search Results for: "${query}"\n\n`;

      if (results.length === 0) {
        formatted += "No results found.\n";
        return formatted;
      }

      for (let i = 0; i < results.length; i++) {
        const r = results[i];
        formatted += `### ${i + 1}. ${r.title || "Untitled"}\n`;
        formatted += `**URL:** ${r.url}\n`;
        const description = r.content || r.snippet || "No description available";
        // Clean up HTML tags from description
        const cleanDescription = description
          .replace(/<[^>]*>/g, "")
          .replace(/&amp;/g, "&")
          .replace(/&lt;/g, "<")
          .replace(/&gt;/g, ">")
          .replace(/&quot;/g, '"')
          .replace(/&#39;/g, "'");
        formatted += `${cleanDescription}\n\n`;
      }

      if (data.suggestions && data.suggestions.length > 0) {
        formatted += `**Related searches:** ${data.suggestions.slice(0, 3).join(", ")}\n`;
      }

      return formatted;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      console.error(`SearchXNG instance ${instance} failed:`, lastError.message);
      // Try next instance
      continue;
    }
  }

  // All instances failed
  return `## Web Search Failed\n\nUnable to perform web search. All search instances are unavailable.\n\nError: ${lastError?.message || "Unknown error"}\n\nPlease try again later.`;
}

/**
 * Search for LLM deployment configurations
 *
 * @param gpuModel - GPU model name (e.g., "RTX 4090", "A100")
 * @param llmModel - LLM model name (e.g., "Qwen2.5-7B", "Llama-3.1-8B")
 * @param backend - Inference backend (e.g., "vLLM", "SGLang")
 * @returns Formatted search results focused on deployment configurations
 */
export async function searchLLMConfig(
  gpuModel: string,
  llmModel: string,
  backend: string = "vLLM"
): Promise<string> {
  // Build optimized search query for LLM deployment configurations
  const queries = [
    `${llmModel} ${gpuModel} ${backend} optimal settings configuration`,
    `${llmModel} deployment ${gpuModel} performance benchmark`,
    `${backend} ${llmModel} tensor parallel gpu memory settings`,
  ];

  let allResults = `## LLM Deployment Configuration Search\n\n`;
  allResults += `**Target Configuration:**\n`;
  allResults += `- GPU: ${gpuModel}\n`;
  allResults += `- Model: ${llmModel}\n`;
  allResults += `- Backend: ${backend}\n\n`;

  // Search with primary query
  const mainResult = await webSearch(queries[0], 5);
  allResults += mainResult;

  return allResults;
}
