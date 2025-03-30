from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session  # Importa la extensión
from main import main
import os, json, glob

# Configuración básica de la aplicación
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Configuración para usar sesiones en el servidor (almacenamiento en filesystem)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'flask_session')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = 'clave_secreta'
Session(app)  # Inicializa la extensión

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
    
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        advanced_mode = request.form.get("advanced_mode") == "true"  # Convertir a booleano

        if user_input:
            # Añadir mensaje del usuario
            session['conversation'].append({
                "sender": translations.get("user", "User"),
                "message": user_input
            })
            
            # Obtener respuesta del bot con el estado avanzado
            bot_response = main(user_input, advanced_mode)
            
            # Añadir respuesta del bot
            session['conversation'].append({
                "sender": translations.get("ai", "AI"),
                "message": bot_response
            })
            
            session.modified = True  # Guardar cambios en la sesión
            
        return redirect(url_for("index"))
    
    return render_template("index.html", 
                           translations=translations,
                           conversation=session.get('conversation', []))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
