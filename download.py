import gdown

files = [
    "1oWcyH0XkzpHeWozMBWJSFEUEw70Lrc2-",
    "1m1SudlRSlEK7y_-jweDjhPB5VVWzmQ7-",
    "1suFO8EBLxRH6hKKcJln4a9PRsOGu2oYj"
]

for idx, f_id in enumerate(files):
    url = f"https://drive.google.com/uc?id={f_id}"
    output = f"doc_{idx+1}.pdf"
    gdown.download(url, output, quiet=False)
