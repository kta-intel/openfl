import torch
import copy
import numpy as np

class PostTransform:
    def __call__(self, x):
        return x
        
class ApplySigmoidPostTransform():
    def __init__(self):
        self.one = torch.tensor(1.0)

    def __call__(self, x):
        output = torch.sigmoid(x)
        return torch.cat([self.one - output, output], dim=1)
    

class AbstractPyTorchTreeImpl(torch.nn.Module):
    """
    Abstract class definig the basic structure for tree-base models implemented in PyTorch.
    """

    def __init__(
        self, tree_parameters, n_features, classes, n_classes, decision_cond="<=", **kwargs
    ):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            n_classes: The total number of used classes
            decision_cond: The condition of the decision nodes in the x <cond> threshold order. Default '<='. Values can be <=, <, >=, >
        """
        super(AbstractPyTorchTreeImpl, self).__init__(**kwargs)

        # Set up the variables for the subclasses.
        # Each subclass will trigger different behaviours by properly setting these.
        self.perform_class_select = False
        self.regression = False ##
        self.classification = False ##
        self.classes = classes

        if classes is None:
            self.regression = True
            self.n_classes = 1 if n_classes is None else n_classes
        else:
            self.classification = True
            self.n_classes = len(classes) if n_classes is None else n_classes
            if min(classes) != 0 or max(classes) != len(classes) - 1:
                self.classes = torch.nn.Parameter(torch.IntTensor(classes), requires_grad=False)
                self.perform_class_select = True

        # Set the decision condition.
        decision_cond_map = {"<=": torch.le, "<": torch.lt, ">=": torch.ge, ">": torch.gt, "=": torch.eq, "!=": torch.ne}
        assert decision_cond in decision_cond_map.keys(), "decision_cond has to be one of:{}".format(
            ",".join(decision_cond_map.keys())
        )
        self.decision_cond = decision_cond_map[decision_cond]

        # In some cases float64 is required oterwise we will lose precision.
        self.tree_op_precision_dtype = "float32"


class GEMMTreeImpl(AbstractPyTorchTreeImpl):
    """
    Class implementing the GEMM strategy in PyTorch for tree-base models.
    """

    def __init__(self, tree_parameters, n_features, classes, n_classes=None, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            n_classes: The total number of used classes
        """
        # If n_classes is not provided we induce it from tree parameters. Multioutput regression targets are also treated as separate classes.
        n_classes = n_classes if n_classes is not None else tree_parameters[0][0][2].shape[0]
        super(GEMMTreeImpl, self).__init__(
            tree_parameters, n_features, classes, n_classes, **kwargs
        )

        # Initialize the actual model.
        hidden_one_size = 0
        hidden_two_size = 0
        hidden_three_size = self.n_classes

        for weight, bias in tree_parameters:
            hidden_one_size = max(hidden_one_size, weight[0].shape[0])
            hidden_two_size = max(hidden_two_size, weight[1].shape[0])

        n_trees = len(tree_parameters)
        weight_1 = np.zeros((n_trees, hidden_one_size, n_features))
        bias_1 = np.zeros((n_trees, hidden_one_size), dtype=np.float64)
        weight_2 = np.zeros((n_trees, hidden_two_size, hidden_one_size))
        bias_2 = np.zeros((n_trees, hidden_two_size))
        weight_3 = np.zeros((n_trees, hidden_three_size, hidden_two_size), dtype=np.float64)

        for i, (weight, bias) in enumerate(tree_parameters):
            if len(weight[0]) > 0:
                weight_1[i, 0 : weight[0].shape[0], 0 : weight[0].shape[1]] = weight[0]
                bias_1[i, 0 : bias[0].shape[0]] = bias[0]
                weight_2[i, 0 : weight[1].shape[0], 0 : weight[1].shape[1]] = weight[1]
                bias_2[i, 0 : bias[1].shape[0]] = bias[1]
                weight_3[i, 0 : weight[2].shape[0], 0 : weight[2].shape[1]] = weight[2]

        self.n_trees = n_trees
        self.n_features = n_features
        self.hidden_one_size = hidden_one_size
        self.hidden_two_size = hidden_two_size
        self.hidden_three_size = hidden_three_size

        self.weight_1 = torch.nn.Parameter(
            torch.from_numpy(weight_1.reshape(-1, self.n_features).astype("float32")).detach().clone()
        )
        self.bias_1 = torch.nn.Parameter(
            torch.from_numpy(bias_1.reshape(-1, 1).astype(self.tree_op_precision_dtype)).detach().clone()
        )

        self.weight_2 = torch.nn.Parameter(torch.from_numpy(weight_2.astype("float32")).detach().clone())
        self.bias_2 = torch.nn.Parameter(torch.from_numpy(bias_2.reshape(-1, 1).astype("float32")).detach().clone())

        self.weight_3 = torch.nn.Parameter(torch.from_numpy(weight_3.astype(self.tree_op_precision_dtype)).detach().clone())

    def forward(self, x):
        x = x.t()
        x = self.decision_cond(torch.mm(self.weight_1, x), self.bias_1)
        x = x.view(self.n_trees, self.hidden_one_size, -1)
        x = x.float()

        x = torch.matmul(self.weight_2, x)

        x = x.view(self.n_trees * self.hidden_two_size, -1) == self.bias_2
        x = x.view(self.n_trees, self.hidden_two_size, -1)
        if self.tree_op_precision_dtype == "float32":
            x = x.float()
        else:
            x = x.double()

        x = torch.matmul(self.weight_3, x)
        x = x.view(self.n_trees, self.hidden_three_size, -1)

        x = self.aggregation(x)

        if self.regression:
            return x

        if self.perform_class_select:
            return torch.index_select(self.classes, 0, torch.argmax(x, dim=1)), x
        else:
            return torch.argmax(x, dim=1), x
        

class GEMMGBDTImpl(GEMMTreeImpl):
    """
    Class implementing the GEMM strategy (in PyTorch) for GBDT models.
    """

    def __init__(self, tree_parameters, n_features, classes=None, **kwargs):
        """
        Args:
            tree_parameters: The parameters defining the tree structure
            n_features: The number of features input to the model
            classes: The classes used for classification. None if implementing a regression model
            extra_config: Extra configuration used to properly implement the source tree
        """
        super(GEMMGBDTImpl, self).__init__(tree_parameters, n_features, classes, **kwargs)

        self.n_gbdt_classes = 1
        self.post_transform = ApplySigmoidPostTransform()

        if classes is not None:
            self.n_gbdt_classes = len(classes) if len(classes) > 2 else 1

        self.n_trees_per_class = len(tree_parameters) // self.n_gbdt_classes

    def aggregation(self, x):
        output = torch.squeeze(x).t().view(-1, self.n_gbdt_classes, self.n_trees_per_class).sum(2)

        return self.post_transform(output)

    def predict(self, X):
        # Ensure the model is in evaluation mode
        self.eval()
        
        # Convert input data to a PyTorch tensor
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Perform the forward pass
        with torch.no_grad():
            output = self.forward(X)[0]
        
        # Convert the output to numpy array and return
        return output.numpy()


# Acquire tree parameters
class TreeParameters:
    """
    Class containing a convenient in-memory representation of a decision tree.
    """

    def __init__(self, lefts, rights, features, thresholds, values):
        """
        Args:
            lefts: The id of the left nodes
            rights: The id of the right nodes
            feature: The features used to make decisions
            thresholds: The thresholds used in the decisions
            values: The value stored in the leaves
        """
        self.lefts = lefts
        self.rights = rights
        self.features = features
        self.thresholds = thresholds
        self.values = values
        
def get_tree_parameters(tree_info):
    """
    Parse the tree and returns an in-memory friendly representation of its structure.
    """
    lefts = []
    rights = []
    features = []
    thresholds = []
    values = []

    tree_traversal(
        tree_info.replace("[f", "").replace("[", "").replace("]", "").split(), lefts, rights, features, thresholds, values
    )

    return TreeParameters(lefts, rights, features, thresholds, values)
    
def tree_traversal(tree_info, lefts, rights, features, thresholds, values):
    """
    Recursive function for parsing a tree and filling the input data structures.
    """
    count = 0
    while count < len(tree_info):
        if "leaf" in tree_info[count]:
            features.append(0)
            thresholds.append(0)
            values.append([float(tree_info[count].split("=")[1])])
            lefts.append(-1)
            rights.append(-1)
            count += 1
        else:
            features.append(int(tree_info[count].split(":")[1].split("<")[0].replace("[f", "")))
            thresholds.append(float(tree_info[count].split(":")[1].split("<")[1].replace("]", "")))
            values.append([-1])
            count += 1
            l_wrong_id = tree_info[count].split(",")[0].replace("yes=", "")
            l_correct_id = 0
            temp = 0
            while not tree_info[temp].startswith(str(l_wrong_id + ":")):
                if "leaf" in tree_info[temp]:
                    temp += 1
                else:
                    temp += 2
                l_correct_id += 1
            lefts.append(l_correct_id)

            r_wrong_id = tree_info[count].split(",")[1].replace("no=", "")
            r_correct_id = 0
            temp = 0
            while not tree_info[temp].startswith(str(r_wrong_id + ":")):
                if "leaf" in tree_info[temp]:
                    temp += 1
                else:
                    temp += 2
                r_correct_id += 1
            rights.append(r_correct_id)

            count += 1

class Node:
    """
    Class defining a tree node.
    """

    def __init__(self, id=None):
        """
        Args:
            id: A unique ID for the node
            left: The id of the left node
            right: The id of the right node
            feature: The feature used to make a decision (if not leaf node, ignored otherwise)
            threshold: The threshold used in the decision (if not leaf node, ignored otherwise)
            value: The value stored in the leaf (ignored if not leaf node).
        """
        self.id = id
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.value = None

def find_max_depth(tree_parameters):
    """
    Function traversing all trees in sequence and returning the maximum depth.
    """
    depth = 0

    for tree in tree_parameters:
        tree = copy.deepcopy(tree)

        lefts = tree.lefts
        rights = tree.rights

        ids = [i for i in range(len(lefts))]
        nodes = list(zip(ids, lefts, rights))

        nodes_map = {0: Node(0)}
        current_node = 0
        for i, node in enumerate(nodes):
            id, left, right = node

            if left != -1:
                l_node = Node(left)
                nodes_map[left] = l_node
            else:
                lefts[i] = id
                l_node = -1

            if right != -1:
                r_node = Node(right)
                nodes_map[right] = r_node
            else:
                rights[i] = id
                r_node = -1

            nodes_map[current_node].left = l_node
            nodes_map[current_node].right = r_node

            current_node += 1

        depth = max(depth, find_depth(nodes_map[0], -1))

    return depth
    
def find_depth(node, current_depth):
    """
    Recursive function traversing a tree and returning the maximum depth.
    """
    if node.left == -1 and node.right == -1:
        return current_depth + 1
    elif node.left != -1 and node.right == -1:
        return find_depth(node.l, current_depth + 1)
    elif node.right != -1 and node.left == -1:
        return find_depth(node.r, current_depth + 1)
    elif node.right != -1 and node.left != -1:
        return max(find_depth(node.left, current_depth + 1), find_depth(node.right, current_depth + 1))

def get_parameters_for_gemm_common(lefts, rights, features, thresholds, values, n_features):
    """
    Common functions used by all tree algorithms to generate the parameters according to the GEMM strategy.

    Args:
        left: The left nodes
        right: The right nodes
        features: The features used in the decision nodes
        thresholds: The thresholds used in the decision nodes
        values: The values stored in the leaf nodes
        n_features: The number of expected input features

    Returns:
        The weights and bias for the GEMM implementation
    """
    values = np.array(values)
    weights = []
    biases = []

    if len(lefts) == 1:
        # Model creating trees with just a single leaf node. We transform it
        # to a model with one internal node.
        lefts = [1, -1, -1]
        rights = [2, -1, -1]
        features = [0, 0, 0]
        thresholds = [0, 0, 0]
        n_classes = values.shape[1]
        values = np.array([np.zeros(n_classes), values.reshape(1), values.reshape(1)])
        values.reshape(3, n_classes)

    # First hidden layer has all inequalities.
    hidden_weights = []
    hidden_biases = []
    for left, feature, thresh in zip(lefts, features, thresholds):
        if left != -1:
            hidden_weights.append([1 if i == feature else 0 for i in range(n_features)])
            hidden_biases.append(thresh)
    weights.append(np.array(hidden_weights).astype("float32"))
    biases.append(np.array(hidden_biases, dtype=np.float64))

    n_splits = len(hidden_weights)

    # Second hidden layer has ANDs for each leaf of the decision tree.
    # Depth first enumeration of the tree in order to determine the AND by the path.
    hidden_weights = []
    hidden_biases = []

    path = [0]
    n_nodes = len(lefts)
    visited = [False for _ in range(n_nodes)]

    class_proba = []
    nodes = list(zip(lefts, rights, features, thresholds, values))

    while True and len(path) > 0:
        i = path[-1]
        visited[i] = True
        left, right, feature, threshold, value = nodes[i]
        if left == -1 and right == -1:
            vec = [0 for _ in range(n_splits)]
            # Keep track of positive weights for calculating bias.
            num_positive = 0
            for j, p in enumerate(path[:-1]):
                num_leaves_before_p = list(lefts[:p]).count(-1)
                if path[j + 1] in lefts:
                    vec[p - num_leaves_before_p] = 1
                    num_positive += 1
                elif path[j + 1] in rights:
                    vec[p - num_leaves_before_p] = -1
                else:
                    raise RuntimeError("Inconsistent state encountered while tree translation.")

            if values.shape[-1] > 1:
                proba = (values[i] / np.sum(values[i])).flatten()
            else:
                # We have only a single value. e.g., GBDT
                proba = values[i].flatten()
            # Some Sklearn tree implementations require normalization.
            class_proba.append(proba)

            hidden_weights.append(vec)
            hidden_biases.append(num_positive)
            path.pop()
        elif not visited[left]:
            path.append(left)
        elif not visited[right]:
            path.append(right)
        else:
            path.pop()

    weights.append(np.array(hidden_weights).astype("float32"))
    biases.append(np.array(hidden_biases).astype("float32"))

    # OR neurons from the preceding layer in order to get final classes.
    weights.append(np.transpose(np.array(class_proba).astype("float64")))
    biases.append(None)

    return weights, biases