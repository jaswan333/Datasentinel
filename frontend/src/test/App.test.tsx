import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { App } from "@/app/App";

describe("DataSentinel Frontend Foundation & Home Page", () => {
  it("renders the primary brand header and navigation items", () => {
    render(<App />);

    // Brand and logo
    expect(screen.getAllByText("DataSentinel").length).toBeGreaterThan(0);

    // Sidebar navigation items scoped to Main Navigation
    const mainNav = screen.getByRole("navigation", { name: /main navigation/i });
    expect(within(mainNav).getByRole("link", { name: /home/i })).toBeInTheDocument();
    expect(within(mainNav).getByRole("link", { name: /machine learning/i })).toBeInTheDocument();
    expect(within(mainNav).getByRole("link", { name: /deep learning/i })).toBeInTheDocument();
    expect(within(mainNav).getByRole("link", { name: /help & about/i })).toBeInTheDocument();
    expect(within(mainNav).getByRole("link", { name: /settings/i })).toBeInTheDocument();
  });

  it("renders the Hero section with accurate titles and action buttons", () => {
    render(<App />);

    // Hero title
    expect(
      screen.getByRole("heading", { name: /trust your training data\./i })
    ).toBeInTheDocument();

    // Hero supporting description
    expect(
      screen.getByText(/investigate suspicious training data with evidence-based/i)
    ).toBeInTheDocument();

    // Primary action buttons
    expect(
      screen.getByRole("button", { name: /explore tabular audits/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /explore image audits/i })
    ).toBeInTheDocument();
  });

  it("renders all four core feature cards", () => {
    render(<App />);

    expect(screen.getByText("Threat Detection")).toBeInTheDocument();
    expect(screen.getByText("Evidence-Based Risk")).toBeInTheDocument();
    expect(screen.getByText("Agentic Investigation")).toBeInTheDocument();
    expect(screen.getByText("Controlled Review")).toBeInTheDocument();
  });

  it("renders the three-step workflow", () => {
    render(<App />);

    expect(screen.getByText("Select Dataset or Demonstration")).toBeInTheDocument();
    expect(screen.getByText("Review Detector Signals & Evidence")).toBeInTheDocument();
    expect(screen.getByText("Investigate Findings & Review Actions")).toBeInTheDocument();
  });

  it("renders the analysis mode entry points", () => {
    render(<App />);

    expect(screen.getByText("Tabular Machine Learning")).toBeInTheDocument();
    expect(screen.getByText("Image Deep Learning")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /open machine learning/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /open deep learning/i })).toBeInTheDocument();
  });

  it("renders the mandatory threat limitations and scientific integrity note", () => {
    render(<App />);

    expect(
      screen.getByText(/detection depends on the threat type, dataset, and available evidence/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/scientific integrity & threat limitations note/i)
    ).toBeInTheDocument();
  });
});
