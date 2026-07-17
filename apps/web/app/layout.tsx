import type { Metadata } from "next";
import { Theme } from "@radix-ui/themes";
import "@radix-ui/themes/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Posy",
  description:
    "Heartfelt thank-you notes in your own handwriting — written with you, printed, and mailed for you.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <Theme
          accentColor="ruby"
          grayColor="sand"
          radius="large"
          appearance="light"
        >
          {children}
        </Theme>
      </body>
    </html>
  );
}
