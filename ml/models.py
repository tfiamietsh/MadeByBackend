import tensorflow as tf
import tensorflow_recommenders as tfrs
import logging

tf.get_logger().setLevel(logging.ERROR)
tf.get_logger().addHandler(logging.FileHandler('tf.log'))


class UserModel(tf.keras.Model):
    def __init__(self, unique_user_ids, embedding_dim):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim)
        ])

    def call(self, ids, training=False, mask=None):
        return self.user_embedding(ids)


class ItemModel(tf.keras.Model):
    def __init__(self, unique_item_titles, embedding_dim, max_tokens):
        super().__init__()

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_item_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_item_titles) + 1, embedding_dim)
        ])
        self.title_text_embedding = tf.keras.Sequential([
            tf.keras.layers.TextVectorization(vocabulary=unique_item_titles, max_tokens=max_tokens),
            tf.keras.layers.Embedding(max_tokens, embedding_dim, mask_zero=True),
            tf.keras.layers.GlobalMaxPool1D()
        ])

    def call(self, titles, training=False, mask=None):
        return tf.concat([self.title_embedding(titles), self.title_text_embedding(titles)], axis=1)


class RetrievalModel(tfrs.models.Model):
    def __init__(self, unique_user_ids, items, unique_item_ids, embedding_dim, max_tokens, k):
        super().__init__()

        self.items = items
        self.index = None
        self.k = k
        self.query_model = tf.keras.Sequential([
            UserModel(unique_user_ids, embedding_dim),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.candidate_model = tf.keras.Sequential([
            ItemModel(unique_item_ids, embedding_dim, max_tokens),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            candidates=items.batch(4).map(self.candidate_model)))

    def call(self, user_id, training=False, mask=None):
        if self.index is None:
            self.index = tfrs.layers.factorized_top_k.BruteForce(self.query_model, self.k)

            self.index.index_from_dataset(
                self.items.batch(4).map(lambda _id: (_id, self.candidate_model(_id))))
        return self.index(user_id)[1]

    def compute_loss(self, features, training=False):
        query_embedding = self.query_model(features['user_id'])
        candidate_embedding = self.candidate_model(features['item_id'])

        return self.task(query_embedding, candidate_embedding)

    def save(self, path):
        tf.saved_model.save(self.index, path)

    @staticmethod
    def load(path):
        return tf.saved_model.load(path)


class RankingModel(tfrs.models.Model):
    def __init__(self, unique_user_ids, unique_item_ids, layer_dims, embedding_dim):
        super().__init__()

        self.user_embedding = UserModel(unique_user_ids, embedding_dim)
        self.item_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_item_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dim)
        ])
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation='relu' if dim != 1 else None) for dim in layer_dims
        ])
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features, training=False, mask=None):
        user_embedding = self.user_embedding(features['user_id'])
        item_embedding = self.item_embedding(features['item_id'])

        return self.ratings(tf.concat([user_embedding, item_embedding], axis=1))

    def compute_loss(self, features, training=False):
        labels = features.pop('item_amount')
        amount_predictions = self(features)

        return self.task(labels=labels, predictions=amount_predictions)

    def save(self, path):
        tf.saved_model.save(self, path)

    @staticmethod
    def load(path):
        return tf.saved_model.load(path)
