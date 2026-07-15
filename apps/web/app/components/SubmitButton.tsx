"use client";

import { useFormStatus } from "react-dom";

// Disables itself while its form's server action is in flight, so a slow
// save can't be double-submitted, and shows the user something happened.
export function SubmitButton({
  children,
  pendingLabel = "Saving…",
  className = "button",
}: {
  children: React.ReactNode;
  pendingLabel?: string;
  className?: string;
}) {
  const { pending } = useFormStatus();
  return (
    <button className={className} type="submit" disabled={pending}>
      {pending ? pendingLabel : children}
    </button>
  );
}
