-- M1: the relationship graph — events, contacts, households, addresses.
-- Shapes per docs/ARCHITECTURE.md: contacts are the durable center;
-- households are first-class with time-tracked membership; addresses are
-- separate from contacts (they change over the years); events select
-- recipients rather than owning them. RLS on every table, no exceptions.

create table public.events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  title text not null,
  occasion_type text not null default 'other'
    check (occasion_type in (
      'wedding', 'bridal_shower', 'baby_shower', 'birthday',
      'graduation', 'holiday', 'other'
    )),
  event_date date,
  user_role text,
  -- V1 only creates past/reactive events; the shape permits future ones.
  temporal_mode text not null default 'past'
    check (temporal_mode in ('past', 'upcoming')),
  recurrence text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.contacts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  full_name text not null,
  relationship text,
  tone_override text,
  format_preferences jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.households (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  name text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.household_members (
  household_id uuid not null references public.households (id) on delete cascade,
  contact_id uuid not null references public.contacts (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  member_from date,
  member_until date,
  primary key (household_id, contact_id)
);

-- One table for addresses; each row belongs to a contact OR a household.
create table public.addresses (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  contact_id uuid references public.contacts (id) on delete cascade,
  household_id uuid references public.households (id) on delete cascade,
  line1 text not null,
  line2 text,
  city text not null,
  state text,
  postal_code text,
  country text not null default 'US',
  raw_input text,
  source text not null default 'manual'
    check (source in ('manual', 'csv', 'contacts', 'parsed_text', 'ocr', 'request_link')),
  validation_status text not null default 'unvalidated'
    check (validation_status in ('unvalidated', 'valid', 'needs_review', 'undeliverable')),
  is_current boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (num_nonnulls(contact_id, household_id) = 1)
);

-- Which people/households an event is thanking. Recipient entry never
-- blocks card-writing: a row here needs only a name behind it.
create table public.event_recipients (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  event_id uuid not null references public.events (id) on delete cascade,
  contact_id uuid references public.contacts (id) on delete cascade,
  household_id uuid references public.households (id) on delete cascade,
  created_at timestamptz not null default now(),
  check (num_nonnulls(contact_id, household_id) = 1),
  unique (event_id, contact_id),
  unique (event_id, household_id)
);

-- RLS: every table, owner-only, both directions.
alter table public.events enable row level security;
alter table public.contacts enable row level security;
alter table public.households enable row level security;
alter table public.household_members enable row level security;
alter table public.addresses enable row level security;
alter table public.event_recipients enable row level security;

create policy "events: own rows" on public.events
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "contacts: own rows" on public.contacts
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "households: own rows" on public.households
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "household_members: own rows" on public.household_members
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "addresses: own rows" on public.addresses
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "event_recipients: own rows" on public.event_recipients
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
