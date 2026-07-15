import Link from "next/link";
import { createEvent } from "../actions";

const OCCASIONS = [
  ["wedding", "Wedding"],
  ["bridal_shower", "Bridal shower"],
  ["baby_shower", "Baby shower"],
  ["birthday", "Birthday"],
  ["graduation", "Graduation"],
  ["holiday", "Holiday"],
  ["other", "Something else"],
] as const;

export default async function NewEventPage({
  searchParams,
}: {
  searchParams: Promise<{
    error?: string;
    message?: string;
    title?: string;
    occasion_type?: string;
    event_date?: string;
  }>;
}) {
  const params = await searchParams;

  return (
    <main className="page">
      <p>
        <Link href="/app">&larr; Back</Link>
      </p>
      <h1 className="wordmark">New event</h1>
      <div className="card stack">
        {params.error && (
          <p className="notice">
            {params.error === "title"
              ? "Give your event a name so you can find it again."
              : `The save didn't go through${
                  params.message ? ` — the database said: “${params.message}”` : ""
                }. Your entries are still filled in below; try again.`}
          </p>
        )}
        <form className="stack" action={createEvent}>
          <label className="stack">
            <span>What&rsquo;s the occasion called?</span>
            <input
              className="input"
              name="title"
              required
              placeholder="Sarah &amp; Tom&rsquo;s Wedding"
              defaultValue={params.title ?? ""}
            />
          </label>
          <label className="stack">
            <span>What kind of occasion?</span>
            <select
              className="input"
              name="occasion_type"
              defaultValue={params.occasion_type ?? "wedding"}
            >
              {OCCASIONS.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="stack">
            <span>When was it? (optional)</span>
            <input
              className="input"
              type="date"
              name="event_date"
              defaultValue={params.event_date ?? ""}
            />
          </label>
          <div>
            <button className="button" type="submit">
              Create event
            </button>
          </div>
        </form>
      </div>
    </main>
  );
}
