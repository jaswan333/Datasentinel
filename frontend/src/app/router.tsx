import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import { HomePage } from "@/pages/HomePage";
import { MachineLearningPage } from "@/pages/MachineLearningPage";
import { DeepLearningPage } from "@/pages/DeepLearningPage";
import { HelpAboutPage } from "@/pages/HelpAboutPage";
import { SettingsPage } from "@/pages/SettingsPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <HomePage />,
      },
      {
        path: "machine-learning",
        element: <MachineLearningPage />,
      },
      {
        path: "deep-learning",
        element: <DeepLearningPage />,
      },
      {
        path: "help",
        element: <HelpAboutPage />,
      },
      {
        path: "settings",
        element: <SettingsPage />,
      },
      {
        path: "*",
        element: <Navigate to="/" replace />,
      },
    ],
  },
]);
