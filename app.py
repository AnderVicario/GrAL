import asyncio
import glob
import json
import os

from flask import Flask, render_template, request, redirect, url_for, session
from markdown import markdown
from werkzeug.utils import secure_filename

from flask_session import Session
from main import get_application
from utils import allowed_file

# Oinarrizko flask konfigurazioa
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Saioak erabiltzeko konfigurazioa
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'flask_session')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = 'clave_secreta'
Session(app)

# Aplikazioaren logika instantziatu
app_logic = get_application()

# Hizkuntzak kargatu
languages = {}
language_list = glob.glob("web/translations/*.json")
for lang in language_list:
    lang_code = os.path.splitext(os.path.basename(lang))[0]
    with open(lang, 'r', encoding='utf8') as file:
        languages[lang_code] = json.load(file)


@app.before_request
def initialize_conversation():
    if 'conversation' not in session:
        session['conversation'] = []


@app.route("/set_lang/<lang>")
def set_lang(lang):
    if lang in languages:
        session['app_language'] = lang
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    lang_code = session.get('app_language', 'en_US')
    translations = languages.get(lang_code, {})

    if request.method == "POST":
        # Modu aurreratua formulariotik lortu
        is_advanced = request.form.get('advanced_mode') == 'true'
        # Saioan gorde
        session['advanced_mode'] = is_advanced

        # Fitxategien prozesatzea
        if 'file' in request.files:
            files = request.files.getlist('file')
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join('data', filename)
                    file.save(filepath)
                    try:
                        asyncio.run(app_logic.process_document(filepath, filename))
                    except Exception as e:
                        print(f"Error processing document: {str(e)}")

        # Erabiltzailearen kontsulta prozesatzea
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            session['conversation'].append({
                "sender": "0",
                "message": user_input
            })

            # Logika nagusira deitu
            bot_reports = app_logic.process_query(user_input, is_advanced)

            # Egiaztatu bot_reports lista den edo ez
            if isinstance(bot_reports, list):
                formatted_reports = [{
                    "entity": report.get("entity_name", "Unknown entity"),
                    "content": markdown(report.get("content", "No content available"), extensions=['nl2br']),
                    "ticker": report.get("ticker", "Unknown")
                } for report in bot_reports]
            else:
                # Erabiltzailearentzat mezu bat sortu bot_reports string bada
                formatted_reports = [{
                    "entity": "System Message",
                    "content": bot_reports,
                    "ticker": "-"
                }]

            session['conversation'].append({
                "sender": "1",
                "message": formatted_reports,
                "type": "multi_report",
                "current_index": 0
            })
            session.modified = True  # Saioaren aldaketak gorde

        return redirect(url_for("index"))

    return render_template("index.html",
                           translations=translations,
                           conversation=session.get('conversation', []),
                           advanced_mode=session.get('advanced_mode', False))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", use_reloader=False)
