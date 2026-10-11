/**
 * DataSentinel API Client
 * Enterprise HTTP wrapper with error normalization and type safety.
 */

export class ApiClientError extends Error {
  public statusCode: number;
  public details: any;

  constructor(message: string, statusCode: number = 500, details?: any) {
    super(message);
    this.name = "ApiClientError";
    this.statusCode = statusCode;
    this.details = details;
  }
}

const getBaseUrl = (): string => {
  // Respect environment variable if set, otherwise rely on Vite's /api proxy
  const envUrl = import.meta.env.VITE_API_URL;
  if (envUrl && typeof envUrl === "string" && envUrl.trim().length > 0) {
    return envUrl.replace(/\/$/, "");
  }
  return "";
};

export async function apiRequest<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const baseUrl = getBaseUrl();
  const url = `${baseUrl}${path.startsWith("/") ? path : `/${path}`}`;

  const defaultHeaders: HeadersInit = {};
  if (!(options.body instanceof FormData)) {
    defaultHeaders["Content-Type"] = "application/json";
  }

  const mergedOptions: RequestInit = {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  };

  let response: Response;
  try {
    response = await fetch(url, mergedOptions);
  } catch (err: any) {
    throw new ApiClientError(
      `Network communication error: ${err?.message || "Failed to connect to DataSentinel backend."}`,
      0,
      err
    );
  }

  let data: any;
  const contentType = response.headers.get("content-type");
  if (contentType && contentType.includes("application/json")) {
    try {
      data = await response.json();
    } catch {
      data = null;
    }
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    const errorDetail =
      (data && typeof data === "object" && (data.detail || data.message || data.error)) ||
      (typeof data === "string" && data) ||
      `Request failed with HTTP status ${response.status} (${response.statusText})`;
    throw new ApiClientError(String(errorDetail), response.status, data);
  }

  return data as T;
}
