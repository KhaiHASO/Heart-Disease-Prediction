from flask import Flask
from routes.home_routes import main

app = Flask(__name__, static_folder='assets')
app.register_blueprint(main)


if __name__ == "__main__":
    app.run(debug=True)
