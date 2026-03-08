import "./globals.css";

export const metadata = {
  title: "PlantGuard AI — Automated Plant Disease Detection",
  description: "AI-powered plant disease detection using Deep Learning and Transfer Learning. Upload a leaf image to get instant diagnosis with Grad-CAM explainability and LLM-powered treatment recommendations.",
  keywords: "plant disease detection, deep learning, transfer learning, ResNet50, Grad-CAM, AI agriculture, PlantVillage",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="bg-grid" />
        {children}
      </body>
    </html>
  );
}
