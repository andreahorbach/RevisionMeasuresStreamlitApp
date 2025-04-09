venv_dir="venv"

if [ ! -d $venv_dir ]; then
  python3.11 -m pip install --upgrade pip
  python3.11 -m pip install --user virtualenv
  python3.11 -m venv $venv_dir
  source $venv_dir/bin/activate
  pip install -r requirements.txt
  python3.11 -m spacy download de_core_news_sm
fi

source $venv_dir/bin/activate


streamlit run streamlit_app.py
