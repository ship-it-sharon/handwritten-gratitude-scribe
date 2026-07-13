import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="page">
      <h1 className="wordmark">Posy</h1>
      <p className="tagline">
        Thank-you notes that sound like you, look like your handwriting, and
        arrive in the mail — without the burden.
      </p>

      <div className="card stack">
        <p>
          Weddings, showers, birthdays: the gratitude is real, the stack of
          blank cards is daunting. Posy helps you write each note in your own
          voice, renders it in a handwriting style that&rsquo;s yours, and
          prints and mails every card — matching envelope included.
        </p>
        <p className="muted">
          This is the walking skeleton of Posy&rsquo;s V1. The real landing
          page arrives with milestone M5.
        </p>
        <div>
          <Link className="button" href="/login">
            Sign in
          </Link>
        </div>
      </div>
    </main>
  );
}
