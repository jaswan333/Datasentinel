import "@testing-library/jest-dom";

// Mock fetch for health checks in tests
globalThis.fetch = async () => {
  return {
    ok: true,
    json: async () => ({
      status: "ok",
      project: "DataSentinel",
      version: "0.8.0",
      groq_model: "openai/gpt-oss-20b",
    }),
  } as Response;
};
 
// Mock scrollIntoView for jsdom
window.HTMLElement.prototype.scrollIntoView = function () {};
