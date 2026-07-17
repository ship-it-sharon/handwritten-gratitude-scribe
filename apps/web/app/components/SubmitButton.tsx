"use client";

import { useFormStatus } from "react-dom";
import { Button } from "@radix-ui/themes";

// Disables itself while its form's server action is in flight, so a slow
// save can't be double-submitted, and shows the user something happened.
export function SubmitButton({
  children,
  pendingLabel = "Saving…",
  variant = "solid",
  size = "2",
}: {
  children: React.ReactNode;
  pendingLabel?: string;
  variant?: "solid" | "soft" | "ghost";
  size?: "1" | "2" | "3";
}) {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} variant={variant} size={size}>
      {pending ? pendingLabel : children}
    </Button>
  );
}
