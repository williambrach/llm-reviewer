## Usage local

### Create env

```bash
python -m venv reviewer
```

### Acitvate env

```bash
source reviewer/bin/activate #linux/mac
```

### Install requirements

```bash
pip install -r requirements.txt
```

### Run

```bash
gradio app.py
```

### Open the Web Browser and Upload Your Paper

Open http://0.0.0.0:7799 and upload your paper + enter OpenAI API key . The feedback will be generated in around ~120 seconds.

You should get the following output:

![demo](/assets/demo.png)

If you encounter any error, please first check the server log and then open an issue.

## Usage Docker

```bash
sudo docker build -t llm-reviewer .
```

```bash
sudo docker images
```

```bash
sudo docker run -d -p 7799:7799 llm-reviewer
```

### Nginx

```
sudo vim /etc/nginx/sites-enabled/llmreview
sudo systemctl reload nginx
sudo systemctl restart nginx
```

