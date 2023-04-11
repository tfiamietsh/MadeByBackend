from app import db
from passlib.hash import pbkdf2_sha256 as sha256


class UserModel(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.SmallInteger, primary_key=True, nullable=False)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def add(self):
        db.session.add(self)
        db.session.commit()

    @classmethod
    def find_by_username(cls, username: str):
        return cls.query.filter_by(username=username).first()

    @staticmethod
    def generate_hash(password):
        return sha256.hash(password)

    @staticmethod
    def verify_hash(password: str, hsh: str):
        return sha256.verify(password, hsh)


class RevokedTokenModel(db.Model):
    __tablename__ = 'revoked_tokens'
    token_id = db.Column(db.SmallInteger, primary_key=True, nullable=False)
    jti = db.Column(db.String(120), nullable=False)

    def add(self):
        db.session.add(self)
        db.session.commit()

    @classmethod
    def is_jti_blacklisted(cls, jti: str):
        return bool(cls.query.filter_by(jti=jti).first())


class ItemModel(db.Model):
    __tablename__ = 'items'
    item_id = db.Column(db.SmallInteger, primary_key=True, nullable=False)
    title = db.Column(db.String(120), nullable=False)
    price = db.Column(db.Float, nullable=False)
    crossed_out_price = db.Column(db.Float)

    def add(self):
        db.session.add(self)
        db.session.commit()

    @classmethod
    def get_all(cls):
        def jsonify(item: ItemModel):
            return {
                'id': item.item_id,
                'title': item.title,
                'price': item.price,
                'crossed_out_price': item.crossed_out_price
            }

        return list(map(lambda x: jsonify(x), sorted(ItemModel.query.all(),
                                                     key=lambda item: item.item_id)))


class PurchaseModel(db.Model):
    __tablename__ = 'purchases'
    purchase_id = db.Column(db.SmallInteger, primary_key=True, nullable=False)
    user_id = db.Column(db.SmallInteger, nullable=False)
    item_id = db.Column(db.SmallInteger, nullable=False)
    amount = db.Column(db.SmallInteger, nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)

    def add(self):
        db.session.add(self)
        db.session.commit()

    @classmethod
    def get_all(cls):
        def jsonify(purchase: PurchaseModel):
            return {
                'user_id': purchase.user_id,
                'item_id': purchase.item_id,
                'item_amount': purchase.amount
            }

        return list(map(lambda x: jsonify(x), PurchaseModel.query.all()))
