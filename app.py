from flask import Flask, render_template, request, redirect, url_for, session
import os, locale, json, glob

# Configuración básica de la aplicación
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'templates'))
app = Flask(__name__, template_folder=template_dir)

app.secret_key = 'clave_secreta'  # Necesaria para usar sesiones

languages = {}

language_list = glob.glob("web/translations/*.json")

for lang in language_list:
    filename = lang.split('\\')
    lang_code = filename[0].split('.')[0].split('/')[2]
    print(lang_code)
    with open(lang, 'r', encoding='utf8') as file:
        languages[lang_code] = json.loads(file.read())
        print(languages)

conversation = []


@app.route("/set_lang/<lang>")
def set_lang(lang):
    print(lang)
    if lang in languages:
        session['app_language'] = lang
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    # Obtener el idioma seleccionado de la sesión o usar por defecto
    lang_code = session.get('app_language', 'en_US')
    translations = languages.get(lang_code, {})

    print(translations)
    
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            # Generar una respuesta placeholder para el bot
            bot_response = f"Bot response (placeholder) for: {user_input}"
            conversation.append({"sender": translations.get("user", "User"), "message": user_input})
            conversation.append({"sender": translations.get("ai", "AI Assistant"), "message": bot_response})
        return redirect(url_for("index"))
    
    return render_template("index.html", conversation=conversation, translations=translations)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
