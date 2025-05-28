import asyncio
import glob
import json
import os

from flask import Flask, render_template, request, redirect, url_for, session
from markdown import markdown
from werkzeug.utils import secure_filename

from agents.document_agent import DocumentAgent, DocumentProcessor, VectorMongoDB
from flask_session import Session
from main import main
from utils import allowed_file

# Oinarrizko flask konfigurazioa
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Saioak erabiltzeko (memoria filesystem-n)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'flask_session')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = 'clave_secreta'
Session(app)  # Hasierazi saioak

advanced_mode = False

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
    print(lang)
    if lang in languages:
        session['app_language'] = lang
    return redirect(url_for("index"))


@app.route('/toggle_advanced', methods=['POST'])
def toggle_advanced():
    global advanced_mode
    advanced_mode = not advanced_mode


@app.route("/", methods=["GET", "POST"])
def index():
    lang_code = session.get('app_language', 'en_US')
    translations = languages.get(lang_code, {})
    user_input = request.form.get("user_input", "").strip()
    advanced_mode = request.form.get("advanced_mode") == "true"

    if request.method == "POST":
        if 'file' in request.files:
            files = request.files.getlist('file')
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join('data', filename)
                    file.save(filepath)
                    try:
                        # PDF parseatu
                        pages = asyncio.run(DocumentProcessor.parse_pdf(filepath))
                        first_page = pages[0] if pages else ""

                        # Entitatea aurkitu
                        agent = DocumentAgent()
                        selected_company = agent.select_financial_entity(
                            filename,
                            first_page
                        )

                        full_text = "\n".join(pages)
                        chunks = DocumentProcessor.chunk_text(full_text)

                        vector_db = VectorMongoDB("global_reports")
                        # vector_db.create_vector_index("global_reports")

                        for i, chunk in enumerate(chunks):
                            doc = {
                                "text": chunk,
                                "metadata": {
                                    "entity": selected_company,
                                    "filename": filename,
                                    "analysis_type": "document",
                                    "chunk_number": i + 1,
                                    "total_chunks": len(chunks),
                                    "source": "DocumentAgent",
                                }
                            }
                            vector_db.add_documents([doc])

                    except Exception as e:
                        print(f"Error procesando documento: {str(e)}")
                        continue

        if user_input:
            # Erabiltzailearen sarrera gorde
            session['conversation'].append({
                "sender": "0",
                "message": user_input
            })

            # Botaren erantzuna lortu
            bot_reports = main(user_input, advanced_mode)

            # Convertir a formato Markdown
            formatted_reports = [{
                "entity": report["entity_name"],
                "content": markdown(report["content"], extensions=['nl2br']),
                "ticker": report["ticker"]
            } for report in bot_reports]

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
                           conversation=session.get('conversation', []))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", use_reloader=False)
