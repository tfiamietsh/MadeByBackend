import numpy as np
import pandas as pd
import tensorflow as tf
from ml.models import RetrievalModel, RankingModel
from app import config


class RecSys:
    def __init__(self, inputs, train_test_ratio=float(config['ml']['train_test_ratio']),
                 batch_size=int(config['ml']['batch_size']),
                 layer_dims=tuple(int(dim) for dim in config['ml']['layer_dims'].split(':')),
                 embedding_dim=int(config['ml']['embedding_dim']),
                 max_tokens=int(config['ml']['max_tokens']),
                 k=int(config['ml']['num_recs'])):
        if inputs is not None:
            purchases_df = pd.DataFrame(inputs[0])
            purchases_df['user_id'] = purchases_df['user_id'].astype('str')
            purchases_df['item_id'] = purchases_df['item_id'].astype('str')
            purchases_df['item_amount'] = purchases_df['item_amount'].astype('float32')
            unique_user_ids = purchases_df['user_id'].unique()

            items_df = pd.DataFrame(inputs[1])
            items_df.rename(columns={'id': 'item_id'}, inplace=True)
            items_df['item_id'] = items_df['item_id'].astype('str')
            unique_item_ids = items_df['item_id'].unique()

            purchases_ds = tf.data.Dataset.from_tensor_slices({
                'user_id': tf.cast(purchases_df['user_id'].values, tf.string),
                'item_id': tf.cast(purchases_df['item_id'].values, tf.string),
                'item_amount': tf.cast(purchases_df['item_amount'].values, tf.float32)
            })
            items_ds = tf.data.Dataset.from_tensor_slices(
                tf.cast(items_df['item_id'].values, tf.string)
            )

            total_num = purchases_df.shape[0]
            train_num = int(train_test_ratio * total_num)
            random_seed = int(config['ml']['random_seed'])
            tf.random.set_seed(random_seed)
            shuffled = purchases_ds.shuffle(total_num, seed=random_seed,
                                            reshuffle_each_iteration=False)
            train = shuffled.take(train_num)
            test = shuffled.skip(train_num).take(total_num - train_num)

            self.retrieval_model = RetrievalModel(unique_user_ids, items_ds, unique_item_ids,
                                                  embedding_dim, max_tokens, k)
            self.ranking_model = RankingModel(unique_user_ids, unique_item_ids,
                                              layer_dims, embedding_dim)
            self.cached_train = train.shuffle(total_num).batch(batch_size).cache()
            self.cached_test = test.batch(2 * batch_size).cache()

    def fit(self, epochs=int(config['ml']['epochs']), optimizer=tf.keras.optimizers.Adagrad,
            learning_rate=0.1):
        self.retrieval_model.compile(optimizer(learning_rate))
        self.retrieval_model.fit(self.cached_train, epochs=epochs)
        self.ranking_model.compile(optimizer(learning_rate))
        self.ranking_model.fit(self.cached_train, epochs=epochs)

    def eval(self):
        self.retrieval_model.evaluate(self.cached_test, return_dict=True)
        self.ranking_model.evaluate(self.cached_test, return_dict=True)

    def __call__(self, user_id, k=6):
        ids, weights = [], []
        item_ids = self.retrieval_model(np.array([user_id]))

        if type(item_ids) == tuple:
            item_ids = item_ids[1]
        for item_id in item_ids.numpy()[0]:
            ids.append(int(item_id.decode('utf-8')))
            weights.append(self.ranking_model(
                {'user_id': np.array([user_id]),
                 'item_id': np.array([item_id.decode('utf-8')])}).numpy()[0][0])
        return ids, weights

    def save(self, path):
        self.retrieval_model.save(path + '/retrieval')
        self.ranking_model.save(path + '/ranking')

    def load(self, path):
        self.retrieval_model = RetrievalModel.load(path + '/retrieval')
        self.ranking_model = RankingModel.load(path + '/ranking')
