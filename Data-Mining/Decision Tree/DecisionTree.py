from IPython.display import Image
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
features = iris['data']
target = iris['target']

decisiontree = DecisionTreeClassifier(random_state=0, max_depth=None,
                                      min_samples_split=2, min_samples_leaf=1,
                                      min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0)


# mentraining model>
model = decisiontree.fit(features, target)

# mengambil sempel oberasiv dan membuat prediksi
# sempel berupa data dimensi kelompok
# fungsi predic() => memriksa kelas yang dimiliki
# fungsi predic_promba > memeriksa probabilitas kelas dari prediksi
observation = [[5, 4, 3, 2]]
model.predict(observation)
model.predict_proba(observation)

# membuat grafik visualisasi DT
dot_data = tree.export_graphviz(decisiontree, out_file=None,
                                feature_names=iris['feature_names'],
                                class_names=iris['target_names'])
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png('iris.png')
