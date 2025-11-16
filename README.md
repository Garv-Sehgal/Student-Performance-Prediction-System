# Student Risk Snapshot (Streamlit)

How to deploy to Streamlit Cloud:

1. Create a GitHub repo and add these files:
   - app.py
   - requirements.txt
   - README.md

2. Commit and push to the `main` branch:
   git init
   git add .
   git commit -m "initial commit"
   git branch -M main
   git remote add origin https://github.com/YOURUSERNAME/YOURREPO.git
   git push -u origin main

3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click "New app" → choose your repo, branch `main`, and `app.py` as the main file → Deploy.
5. Wait for the build; your app will be live at the provided URL.

Notes:
- Use sample dataset with a 'Total_Score' column for quick demo.
- Avoid large files in the repo. Uploaded CSVs are processed at runtime and are ephemeral.
