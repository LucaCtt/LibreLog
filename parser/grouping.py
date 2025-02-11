import re
import pandas as pd
import hashlib
from datetime import datetime


class LogCluster:
    """
    A class used to represent a cluster of log entries.

    Attributes
    ----------
    log_template : str
        A template string representing the log pattern.
    log_ids : list
        A list of log entry IDs that belong to this cluster.

    Methods
    -------
    __init__(log_template="", log_ids=None)
        Initializes the LogCluster with a log template and an optional list of log IDs.
    """
    def __init__(self, log_template="", log_ids=None):
        self.log_template = log_template

        if log_ids is None:
            log_ids = []
        self.log_ids = log_ids


class Node:
    """
    A class used to represent a Node in a tree structure.

    Attributes
    ----------
    child_nodes : dict
        A dictionary containing the children nodes.
    depth : int
        The depth of the node in the tree.
    digit_or_token : any
        The value stored in the node, which can be a digit or a token.

    Methods
    -------
    __init__(child_nodes=None, depth=0, digitOrtoken=None)
        Initializes the Node with optional children, depth, and value.
    """
    def __init__(self, child_nodes=None, depth=0, digit_or_token=None):
        if child_nodes is None:
            child_nodes = dict()
        self.child_nodes = child_nodes
        self.depth = depth
        self.digit_or_token = digit_or_token


class LogGrouper:
    def __init__(
        self,
        depth=4,
        similarity_threshold=0.5,
        max_children=100,
        regexes=[],
        keep_params=False,
    ):
        self.total_time = 0.0
        self.depth = depth - 2
        self.similarity_treshold = similarity_threshold
        self.max_children = max_children
        self.log_name = None
        self.df_log = None
        self.regexes = regexes
        self.keep_params = keep_params

    def group(self, logs):
        """
        Groups log messages into clusters based on their content similarity.

        This method processes a list of log messages, groups them into clusters
        based on their content similarity, and returns the clustering result.
        It uses a prefix tree (trie) to efficiently search for matching clusters
        and updates the clusters as new log messages are processed.

        Args:
            logs (list): A list of log messages to be grouped.

        Returns:
            list: A list of clustered log messages.

        """
        start_time = datetime.now()
        root_node = Node()
        log_clusters = []

        self.load_data(logs)

        for _, line in self.df_log.iterrows():
            log_id = line["LineId"]
            log_message = self.__preprocess_line(line["Content"]).strip().split()
            match_cluster = self.__tree_search(root_node, log_message)

            if match_cluster is None:
                new_cluster = LogCluster(log_template=log_message, log_ids=[log_id])
                log_clusters.append(new_cluster)
                self.__add_seq_to_prefix_tree(root_node, new_cluster)

            else:
                new_template = self.__get_template(
                    log_message, match_cluster.log_template
                )
                match_cluster.log_ids.append(log_id)
                if " ".join(new_template) != " ".join(match_cluster.log_template):
                    match_cluster.log_template = new_template

        list_result = self.outputResult(log_clusters)
        time_taken = datetime.now() - start_time
        self.total_time += time_taken.total_seconds()

        return list_result

    def load_data(self, logs):
        """
        Loads and preprocesses log data into a pandas DataFrame.
        
        Args:
            logs (list): List of log messages to be processed.
        """
        def preprocess(log):
            for current_regex in self.regexes:
                log = re.sub(current_regex, "<*>", log)
            return log

        line_count = len(logs)
        self.df_log = pd.DataFrame(logs, columns=["Content"])
        self.df_log.insert(0, "LineId", None)
        self.df_log["LineId"] = [i + 1 for i in range(line_count)]
        self.df_log["Content_"] = self.df_log["Content"].map(preprocess)

    def __preprocess_line(self, line):
        """
        Preprocesses a given line by replacing all substrings that match any of the
        regular expressions in self.regexes with the placeholder "<*>".
        Args:
            line (str): The input line to be preprocessed.
        Returns:
            str: The preprocessed line with matched substrings replaced by "<*>".
        """
        for current_regex in self.regexes:
            line = re.sub(current_regex, "<*>", line)
        return line

    def __tree_search(self, root_node, sequence):
        """
        Searches for the best matching log cluster in the prefix tree for a given log sequence.

        Args:
            root_node (Node): The root node of the prefix tree.
            sequence (list): The log char sequence to be matched.

        Returns:
            LogCluster: The best matching log cluster or None if no match is found.
        """
        best_log_cluster = None

        sequence_length = len(sequence)
        if sequence_length not in root_node.child_nodes:
            return best_log_cluster

        parent_node = root_node.child_nodes[sequence_length]

        current_depth = 1
        for token in sequence:
            if current_depth >= self.depth or current_depth > sequence_length:
                break

            if token in parent_node.child_nodes:
                parent_node = parent_node.child_nodes[token]
            elif "<*>" in parent_node.child_nodes:
                parent_node = parent_node.child_nodes["<*>"]
            else:
                return best_log_cluster
            current_depth += 1

        log_cluster_list = parent_node.child_nodes

        best_log_cluster = self.__fast_match(log_cluster_list, sequence)

        return best_log_cluster

    def __fast_match(self, log_clusters, sequence):
        """
        Finds the best matching log cluster for a given log sequence based on similarity.

        Args:
            log_clusters (list): List of log clusters to match against.
            sequence (list): The log sequence to be matched.

        Returns:
            LogCluster: The best matching log cluster or None if no match is found.
        """
        best_log_cluster = None

        max_similarity = -1
        max_num_of_params = -1
        best_cluster = None

        for log_cluster in log_clusters:
            current_similarity, current_num_of_params = self.__seq_dist(
                log_cluster.log_template, sequence
            )

            if current_similarity > max_similarity or (
                current_similarity == max_similarity
                and current_num_of_params > max_num_of_params
            ):
                max_similarity = current_similarity
                max_num_of_params = current_num_of_params
                best_cluster = log_cluster

        if max_similarity >= self.similarity_treshold:
            best_log_cluster = best_cluster

        return best_log_cluster

    def __seq_dist(self, seq1, seq2):
        """
        Calculates the similarity and number of parameters between two sequences.

        Args:
            seq1 (list): The first sequence to compare.
            seq2 (list): The second sequence to compare.

        Returns:
            tuple: A tuple containing the similarity ratio (float) and the number of parameters (int).
        """
        assert len(seq1) == len(seq2)
        similar_tokens = 0
        num_of_params = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == "<*>":
                num_of_params += 1
                continue
            if token1 == token2:
                similar_tokens += 1

        similarity_ratio = float(similar_tokens) / len(seq1)

        return similarity_ratio, num_of_params

    def __add_seq_to_prefix_tree(self, root_node, log_cluster):
        """
        Adds a log sequence to the prefix tree.

        Args:
            root_node (Node): The root node of the prefix tree.
            log_cluster (LogCluster): The log cluster to be added to the tree.
        """
        sequence_length = len(log_cluster.log_template)
        if sequence_length not in root_node.child_nodes:
            first_layer_node = Node(depth=1, digit_or_token=sequence_length)
            root_node.child_nodes[sequence_length] = first_layer_node
        else:
            first_layer_node = root_node.child_nodes[sequence_length]

        parent_node = first_layer_node

        current_depth = 1
        for token in log_cluster.log_template:
            if current_depth >= self.depth or current_depth > sequence_length:
                if len(parent_node.child_nodes) == 0:
                    parent_node.child_nodes = [log_cluster]
                else:
                    parent_node.child_nodes.append(log_cluster)
                break

            if token not in parent_node.child_nodes:
                if not self.__has_numbers(token):
                    if "<*>" in parent_node.child_nodes:
                        if len(parent_node.child_nodes) < self.max_children:
                            new_node = Node(depth=current_depth + 1, digit_or_token=token)
                            parent_node.child_nodes[token] = new_node
                            parent_node = new_node
                        else:
                            parent_node = parent_node.child_nodes["<*>"]
                    else:
                        if len(parent_node.child_nodes) + 1 < self.max_children:
                            new_node = Node(depth=current_depth + 1, digit_or_token=token)
                            parent_node.child_nodes[token] = new_node
                            parent_node = new_node
                        elif len(parent_node.child_nodes) + 1 == self.max_children:
                            new_node = Node(depth=current_depth + 1, digit_or_token="<*>")
                            parent_node.child_nodes["<*>"] = new_node
                            parent_node = new_node
                        else:
                            parent_node = parent_node.child_nodes["<*>"]

                else:
                    if "<*>" not in parent_node.child_nodes:
                        new_node = Node(depth=current_depth + 1, digit_or_token="<*>")
                        parent_node.child_nodes["<*>"] = new_node
                        parent_node = new_node
                    else:
                        parent_node = parent_node.child_nodes["<*>"]

            else:
                parent_node = parent_node.child_nodes[token]

            current_depth += 1

    def __has_numbers(self, s):
        """
        Check if the given string contains any numeric characters.
        Args:
            s (str): The string to be checked for numeric characters.
        Returns:
            bool: True if the string contains at least one numeric character, False otherwise.
        """
        return any(char.isdigit() for char in s)

    def __get_template(self, seq1, seq2):
        """
        Generates a template by comparing two sequences and replacing differing tokens with a placeholder.

        Args:
            seq1 (list): The first sequence to compare.
            seq2 (list): The second sequence to compare.

        Returns:
            list: A list representing the template with differing tokens replaced by "<*>".
        """
        assert len(seq1) == len(seq2)
        template = []

        for i, word in enumerate(seq1):
            if word == seq2[i]:
                template.append(word)
            else:
                template.append("<*>")

        return template

    def output_result(self, log_clusters):
        """
        Generates the output result from the log clusters.

        Args:
            log_clusters (list): List of log clusters.

        Returns:
            list: List of lists containing log content, event ID, and event template.
        """
        log_templates = [0] * self.df_log.shape[0]
        log_template_ids = [0] * self.df_log.shape[0]
        df_events = []
        for log_cluster in log_clusters:
            template_str = " ".join(log_cluster.log_template)
            occurrence = len(log_cluster.log_ids)
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]
            for log_id in log_cluster.log_ids:
                log_id -= 1
                log_templates[log_id] = template_str
                log_template_ids[log_id] = template_id
            df_events.append([template_id, template_str, occurrence])

        self.df_log["EventId"] = log_template_ids
        self.df_log["EventTemplate"] = log_templates
        if self.keep_params:
            self.df_log["ParameterList"] = self.df_log.apply(
                self.get_parameter_list, axis=1
            )
        array_result = self.df_log.loc[
            :, ["Content", "EventId", "EventTemplate"]
        ].values
        list_result = [list(row) for row in array_result]
        return list_result

    def get_parameter_list(self, row):
        """
        Extracts the parameter list from a log message based on the event template.

        Args:
            row (pd.Series): A row from the log DataFrame containing 'Content' and 'EventTemplate'.

        Returns:
            list: A list of parameters extracted from the log message.
        """
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
        template_regex = re.sub(r"\\ +", r"\s+", template_regex)
        template_regex = "^" + template_regex.replace("<*>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = (
            list(parameter_list)
            if isinstance(parameter_list, tuple)
            else [parameter_list]
        )
        return parameter_list
