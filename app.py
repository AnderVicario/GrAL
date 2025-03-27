from flask import Flask, render_template, request, redirect, url_for, session
from flask_babel import Babel, gettext
import os

# Configuración básica de la aplicación
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'web', 'templates'))
app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'tu_clave_secreta_aqui'  # Necesario para usar sesiones

# Configuración de Babel
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['LANGUAGES'] = {
    'en': 'English',
    'eu': 'Euskara'
}
babel = Babel(app)

conversation = []

@app.route("/", methods=["GET", "POST"])
def index():
    global conversation
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            # Generar una respuesta placeholder para el bot
            bot_response = f"Bot response (placeholder) for: {user_input}"
            conversation.append({"sender": "User", "message": user_input})
            conversation.append({"sender": "Bot", "message": bot_response})
        return redirect(url_for("index"))
    return render_template("index.html", conversation=conversation)

@app.route('/set_lang/<lang>')
def set_lang(lang):
    if lang in app.config['LANGUAGES']:
        session['lang'] = lang
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
