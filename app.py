from flask import Flask
from flask_jwt_extended import JWTManager
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from configparser import ConfigParser

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)
config = ConfigParser()

config.read('config.cfg')
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://{username}:{password}" \
    "@localhost:{port}/{database}".format(username=config['api']['pg_user'],
                                          password=config['api']['pg_password'],
                                          port=config['api']['pg_port'],
                                          database=config['api']['pg_database'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = config['api']['secret_key']
db = SQLAlchemy(app)


@app.before_first_request
def create_tables():
    db.create_all()


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:4200')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


app.config['JWT_SECRET_KEY'] = 'jwt-' + config['api']['secret_key']
jwt = JWTManager(app)
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']


@jwt.token_in_blocklist_loader
def check_if_token_in_blacklist(_, decrypted_token):
    jti = decrypted_token['jti']
    return RevokedTokenModel.is_jti_blacklisted(jti)


from api.resources import *

api.add_resource(UserLogin, '/login')
api.add_resource(UserLogoutAccess, '/logout/access')
api.add_resource(UserLogoutRefresh, '/logout/refresh')
api.add_resource(TokenRefresh, '/token/refresh')

api.add_resource(Items, '/items')
api.add_resource(Purchases, '/purchases')
api.add_resource(Recommendations, '/recommendations')
