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
  searchParams: Promise<{ error?: string }>;
}) {
  const { error } = await searchParams;

  return (
    <main className="page">
      <p>
        <Link href="/app">&larr; Back</Link>
      </p>
      <h1 className="wordmark">New event</h1>
      <div className="card stack">
        {error && (
          <p className="notice">
            {error === "title"
              ? "Give your event a name so you can find it again."
              : "Something went wrong saving — try again."}
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
            />
          </label>
          <label className="stack">
            <span>What kind of occasion?</span>
            <select className="input" name="occasion_type" defaultValue="wedding">
              {OCCASIONS.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="stack">
            <span>When was it? (optional)</span>
            <input className="input" type="date" name="event_date" />
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
