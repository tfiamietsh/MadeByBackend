from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
from flask_restful import Resource, reqparse
from flask import request
from app import db
from api.models import *
from ml.recsys import RecSys
import os
from app import config

login_parser = reqparse.RequestParser()
login_parser.add_argument('username', required=True)
login_parser.add_argument('password', required=True)


class UserLogin(Resource):
    def post(self):
        data = login_parser.parse_args()
        user = UserModel.find_by_username(data['username'])

        if not user:
            return {'message': 'Пользователя \'{}\' не существует'.format(data['username'])}
        if UserModel.verify_hash(data['password'], user.password_hash):
            return {
                'id': user.user_id,
                'username': user.username,
                'access_token': create_access_token(identity=data['username']),
                'refresh_token': create_refresh_token(identity=data['username'])
            }
        else:
            return {'message': 'Неверный пароль'}


class UserLogoutAccess(Resource):
    @jwt_required()
    def post(self):
        jti = get_jwt()['jti']

        try:
            revoked_token = RevokedTokenModel(jti=jti)

            revoked_token.add()
            return {'message': 'Токен был аннулирован'}
        except():
            return {'message': 'Что-то пошло не так'}, 500


class UserLogoutRefresh(Resource):
    @jwt_required(refresh=True)
    def post(self):
        jti = get_jwt()['jti']

        try:
            revoked_token = RevokedTokenModel(jti=jti)

            revoked_token.add()
            return {'message': 'Токен был аннулирован'}
        except():
            return {'message': 'Что-то пошло не так'}, 500


class TokenRefresh(Resource):
    @jwt_required(refresh=True)
    def post(self):
        return {'access_token': create_access_token(identity=get_jwt_identity())}


class Items(Resource):
    @jwt_required()
    def get(self):
        return ItemModel.get_all()


class Purchases(Resource):
    @jwt_required()
    def get(self):
        def jsonify(x):
            return {
                'id': x.ItemModel.item_id,
                'title': x.ItemModel.title,
                'amount': x.PurchaseModel.amount,
                'date': str(x.PurchaseModel.date),
                'time': str(x.PurchaseModel.time)
            }

        query = db.session.query(ItemModel, PurchaseModel) \
            .filter(ItemModel.item_id == PurchaseModel.item_id) \
            .filter(PurchaseModel.user_id == request.args['user_id'])
        records = query.all()

        return list(map(lambda x: jsonify(x),
                        sorted(records, key=lambda purchase:
                        (purchase.PurchaseModel.date,
                         purchase.PurchaseModel.time))[::-1]))


class Recommendations(Resource):
    @jwt_required()
    def get(self):
        def jsonify(item: ItemModel):
            return {
                'id': item.item_id,
                'title': item.title,
                'price': item.price,
                'crossed_out_price': item.crossed_out_price
            }
        path, recs = os.getcwd() + config['ml']['path'], None

        if os.path.exists(path):
            recsys = RecSys(None)
            recsys.load(path)
            ids, weights = recsys(request.args['user_id'])
        else:
            recsys = RecSys([PurchaseModel.get_all(), ItemModel.get_all()])
            recsys.fit()
            recsys.eval()
            ids, weights = recsys(request.args['user_id'])
            recsys.save(path)

        sorted_ids = [pair[0] for pair in sorted(
            zip(ids, weights), key=lambda pair: pair[1], reverse=True
        )]
        return list(map(lambda x: jsonify(x), db.session.query(ItemModel)
                        .filter(ItemModel.item_id.in_(sorted_ids)).all()))
