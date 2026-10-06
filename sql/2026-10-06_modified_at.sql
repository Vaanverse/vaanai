-- VaanAI: water_insights — one row per location per day
-- Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
-- Safe to re-run.

-- 1) New column: when the row was last written by the agent.
ALTER TABLE public.water_insights
  ADD COLUMN IF NOT EXISTS modified_at timestamptz;

-- Back-fill existing rows so the column isn't empty for old data.
UPDATE public.water_insights
   SET modified_at = created_at
 WHERE modified_at IS NULL;

-- New rows get a value even if something other than the script inserts them.
ALTER TABLE public.water_insights
  ALTER COLUMN modified_at SET DEFAULT now();

-- 2) Make sure created_at is filled in automatically on insert
--    (the script relies on the database setting it).
ALTER TABLE public.water_insights
  ALTER COLUMN created_at SET DEFAULT now();

-- 3) Drop the old UNIQUE rule on (location, before_date, after_date).
--    Without this, next week's run would be REJECTED whenever the agent picks
--    the same two scene dates as an earlier week (no newer clear scene yet).
--    The name of the rule isn't known, so this finds it by its columns.
DO $$
DECLARE r record;
BEGIN
  -- constraints (created with UNIQUE (...) or ADD CONSTRAINT ... UNIQUE)
  FOR r IN
    SELECT con.conname
      FROM pg_constraint con
     WHERE con.conrelid = 'public.water_insights'::regclass
       AND con.contype = 'u'
       AND (SELECT array_agg(att.attname::text ORDER BY att.attname::text)
              FROM unnest(con.conkey) k
              JOIN pg_attribute att
                ON att.attrelid = con.conrelid AND att.attnum = k)
           = ARRAY['after_date','before_date','location']
  LOOP
    EXECUTE format('ALTER TABLE public.water_insights DROP CONSTRAINT %I', r.conname);
    RAISE NOTICE 'Dropped constraint %', r.conname;
  END LOOP;

  -- plain unique indexes (created with CREATE UNIQUE INDEX ...)
  FOR r IN
    SELECT i.relname
      FROM pg_index ix
      JOIN pg_class i ON i.oid = ix.indexrelid
     WHERE ix.indrelid = 'public.water_insights'::regclass
       AND ix.indisunique AND NOT ix.indisprimary
       AND (SELECT array_agg(att.attname::text ORDER BY att.attname::text)
              FROM unnest(ix.indkey) k
              JOIN pg_attribute att
                ON att.attrelid = ix.indrelid AND att.attnum = k)
           = ARRAY['after_date','before_date','location']
  LOOP
    EXECUTE format('DROP INDEX IF EXISTS public.%I', r.relname);
    RAISE NOTICE 'Dropped index %', r.relname;
  END LOOP;
END $$;

-- 4) Fast lookup for "today's row for this location" (what the script checks).
CREATE INDEX IF NOT EXISTS water_insights_location_created_at_idx
  ON public.water_insights (location, created_at DESC);

-- Check: should list modified_at with type "timestamp with time zone".
SELECT column_name, data_type, column_default
  FROM information_schema.columns
 WHERE table_schema = 'public' AND table_name = 'water_insights'
 ORDER BY ordinal_position;
