# copy_images.py
from supabase import create_client

BUCKET = "insight-images"
old = create_client("https://hzsamzgpyztrhhywdxrp.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh6c2FtemdweXp0cmhoeXdkeHJwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5MDUxNDQxNywiZXhwIjoyMTA2MDkwNDE3fQ.K5GI9GGKr14MgFPbi0TMhUu_Aion4bcSM_padqfX-xg")
new = create_client("https://yucfgjnkyebvwywxekqq.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl1Y2Znam5reWVidnd5d3hla3FxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5MDkyMjE3MSwiZXhwIjoyMTA2NDk4MTcxfQ.oCghmwFz_MKN78MM4TxibRap68dAMCKscSCNdDu34p8")

files = old.storage.from_(BUCKET).list("", {"limit": 1000})
for f in files:
    name = f["name"]
    if name.startswith("."):          # skip Supabase's placeholder file
        continue
    data = old.storage.from_(BUCKET).download(name)
    new.storage.from_(BUCKET).upload(name, data, {"content-type": "image/png", "upsert": "true"})
    print("copied", name)
print("done:", len(files), "files")