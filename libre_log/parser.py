from ollama import chat
import re
import random
import textdistance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime


def replace_bracketed_uppercase(text):
    """
    Replace all uppercase bracketed words with '<*>'.

    Args:
        text (str): The input text containing bracketed uppercase words.

    Returns:
        str: The text with bracketed uppercase words replaced by '<*>'.
    """
    pattern = r"<[A-Z_]+>"
    replaced_text = re.sub(pattern, "<*>", text)
    return replaced_text.strip()


def get_logs_from_group(group_list):
    """
    Extract logs from a list of group dictionaries.

    Args:
        group_list (list): A list of dictionaries containing log information.

    Returns:
        list: A list of log contents extracted from the group dictionaries.
    """
    logs_from_group = []
    for ele in group_list:
        logs_from_group.append(ele["Content"])
    return logs_from_group


def verify_one_regex(log, regex):
    """
    Verify if a log matches a given regex pattern.

    Args:
        log (str): The log content.
        regex (str): The regex pattern to match against the log.

    Returns:
        bool: True if the log matches the regex, False otherwise.
    """
    log = log.replace(",", "")
    regex = regex.replace(",", "")
    try:
        if re.search(regex, log):
            return True
        else:
            return False
    except re.error:
        return False


def verify_one_regex_to_match_whole_log(log, regex):
    """
    Verify if a log matches a given regex pattern exactly.

    Args:
        log (str): The log content.
        regex (str): The regex pattern to match against the log.

    Returns:
        bool: True if the log matches the regex exactly, False otherwise.
    """
    log = log.replace(",", "")
    regex = regex.replace(",", "")
    regex = f"^{regex}$"
    try:
        if re.search(regex, log):
            return True
        else:
            return False
    except re.error:
        return False


def check_and_truncate_regex(pattern):
    """
    Truncate a regex pattern if it contains more than 30 wildcards.

    Args:
        pattern (str): The regex pattern to check and truncate.

    Returns:
        str: The truncated regex pattern if it contains more than 30 wildcards, otherwise the original pattern.
    """
    parts = re.split(r"(\(\.\*\?\))", pattern)
    wildcards_count = parts.count("(.*?)")

    if wildcards_count > 30:
        index_30th = [i for i, part in enumerate(parts) if part == "(.*?)"][29]
        truncated_parts = parts[: index_30th + 1]
        truncated_pattern = "".join(truncated_parts)
        return truncated_pattern
    else:
        return pattern


class LogParser:
    def __init__(
        self,
        regex_manager,
        model="llama3-8b",
        regex_sample=5,
        similarity_measure="jaccard",
        do_self_reflection=True,
    ):
        self.total_time = 0.0
        self.new_event = 0
        self.model = model
        self.regex_sample = regex_sample
        self.regex_manager = regex_manager
        self.similarity_measure = similarity_measure
        self.do_self_reflection = do_self_reflection

    def parse(self, groups_from_parser, logs):
        """
        Parse logs and generate regex patterns for them.

        Args:
            groups_from_parser (list): A list of dictionaries containing log information.
            logs (list): A list of log strings.

        Returns:
            list: A list of tuples containing log content, event ID, and regex pattern.
        """
        result_list = []
        start_time = datetime.now()

        if self.__check_predefined_logs(log_list=logs, is_dict=False):
            result_list = self.__store_regex_for_logs(
                result_list, groups_from_parser, re.escape(logs[0]).replace("\ ", " ")
            )
        else:
            for log in logs[::-1]:
                matched_regex = self.regex_manager.find_matched_regex_template(log)
                if matched_regex:
                    logs.remove(log)
                    groups_from_parser = self.__remove_first_matching_item(
                        groups_from_parser, log
                    )
                    result_list.append([log, "0", matched_regex])
            if logs:
                log_regex = self.__generate_log_template(log_list=logs)
                result_list = self.__store_regex_for_logs(
                    result_list, groups_from_parser, log_regex
                )
        time_taken = datetime.now() - start_time
        self.total_time += time_taken.total_seconds()

        return result_list

    def __check_predefined_logs(self, log_list, is_dict=False):
        """
        Check if the log list contains pre-defined logs.

        Args:
            log_list (list): A list of log strings or dictionaries containing logs.
            is_dict (bool, optional): Flag to indicate if logs are in dictionary format. Defaults to False.

        Returns:
            bool: True if the log list contains pre-defined logs, False otherwise.
        """
        if is_dict:
            log_list = get_logs_from_group(log_list)
        unique_logs = list(set(log_list))
        first_log = unique_logs[0]
        if len(unique_logs) != 1:
            return False
        elif (
            (" is " in first_log)
            or ("=" in first_log)
            or (" to " in first_log)
            or ("_" in first_log)
            or ("-" in first_log)
            or (":" in first_log)
            or ("." in first_log)
            or any(char.isdigit() for char in first_log)
        ):
            return False
        return True

    def __store_regex_for_logs(self, result_list, group_dict_list, log_regex):
        """
        Store regex patterns for logs and handle wrong logs.

        Args:
            result_list (list): A list to store the results.
            group_dict_list (list): A list of dictionaries containing log information.
            log_regex (str): The regex pattern to match against the logs.

        Returns:
            list: The updated result list.
        """
        result_list, wrong_logs = self.__check_regex_from_groups(
            result_list, group_dict_list, log_regex
        )
        len_wrong = len(wrong_logs)
        test_time = 0
        while len(wrong_logs) > 0 and (test_time < 3 and len_wrong == len(wrong_logs)):
            len_wrong = len(wrong_logs)
            test_time = test_time + 1
            log_regex = self.__generate_log_template(log_list=wrong_logs, is_dict=True)
            result_list, wrong_logs = self.__check_regex_from_groups(
                result_list, wrong_logs, log_regex, self.new_event
            )
            if len_wrong != len(wrong_logs):
                self.new_event = self.new_event + 1
        for log in wrong_logs:
            result_list.append((log["Content"], self.new_event, log_regex))
            self.new_event = self.new_event + 1
        return result_list

    def __generate_log_template(
        self,
        log_list,
        is_dict=False,
    ):
        """
        Generate a log template using the specified pipeline.

        Args:
            log_list (list): A list of log strings or dictionaries containing logs.
            is_dict (bool, optional): Flag to indicate if logs are in dictionary format. Defaults to False.
            do_sample (bool, optional): Flag to indicate if sampling should be used. Defaults to False.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 1024.

        Returns:
            str: The generated log template.
        """
        if self.__check_predefined_logs(log_list, is_dict=is_dict):
            if is_dict:
                log_list = get_logs_from_group(log_list)
            return re.escape(log_list[0]).replace("\ ", " ")

        prompt, sampled_log_list = self.__generate_prompt_with_log_list(
            log_list, is_dict=is_dict
        )

        response = chat(model=self.model, messages=prompt)
        out = response.message.content
        result = self.__clean_regex(sampled_log_list[0], self.__template_to_regex(out))
        return result

    def __generate_prompt_with_log_list(self, log_list, dic=False):
        """
        Generate a prompt for the log list using the default model.

        Args:
            log_list (list): A list of log strings or dictionaries containing logs.
            dic (bool, optional): Flag to indicate if logs are in dictionary format. Defaults to False.

        Returns:
            tuple: A tuple containing the full prompt and the trimmed list of logs.
        """
        trimmed_log_list = self.__adaptive_random_sampling(
            log_list, self.regex_sample, dic=dic
        )
        messages = [
            {
                "role": "system",
                "content": "You will be provided with a list of logs. You must identify and abstract all the dynamic variables in logs with '<*>' and output ONE static log template that matches all the logs. Print the input logs' template delimited by backticks",
            },
            {
                "role": "user",
                "content": 'Log list: ["try to connected to host: 172.16.254.1, finished.", "try to connected to host: 173.16.254.2, finished."]',
            },
            {
                "role": "assistant",
                "content": "`try to connected to host: <*>, finished.`",
            },
            {"role": "user", "content": f"Log list: {trimmed_log_list}"},
        ]

        return messages, trimmed_log_list

    def cosine_similarity_distance(self, text1, text2):
        """
        Calculate the cosine similarity distance between two texts.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The cosine similarity distance between the two texts.
        """
        vectorizer = CountVectorizer()
        text1_vector = vectorizer.fit_transform([text1])
        text2_vector = vectorizer.transform([text2])
        similarity = cosine_similarity(text1_vector, text2_vector)[0][0]
        return 1 - similarity

    def jaccard_distance(self, text1, text2):
        """
        Calculate the Jaccard distance between two texts.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The Jaccard distance between the two texts.
        """
        return textdistance.jaccard.normalized_distance(text1.split(), text2.split())

    def min_distance(self, candidate_set, target_set):
        """
        Calculate the minimum distance between each candidate in the candidate set and the target set.

        Args:
            candidate_set (list): A list of candidate strings.
            target_set (list): A list of target strings.

        Returns:
            list: A list of minimum distances for each candidate.
        """
        distances = []
        for candidate in candidate_set:
            min_distance = float("inf")
            for target in target_set:
                if self.similarity_measure == "cosine":
                    min_distance = min(
                        min_distance,
                        self.cosine_similarity_distance(candidate, target),
                    )
                elif self.similarity_measure == "jaccard":
                    min_distance = min(
                        min_distance, self.jaccard_distance(candidate, target)
                    )
                else:
                    raise ValueError("Invalid similarity metric.")
            distances.append(min_distance)
        return distances

    def __adaptive_random_sampling(
        self, logs, sample_size, max_logs=200, similarity_flag=False, dic=False
    ):
        """
        Perform adaptive random sampling on a list of logs.

        Args:
            logs (list): A list of log strings or dictionaries containing logs.
            sample_size (int): The number of samples to select.
            max_logs (int, optional): The maximum number of logs to consider. Defaults to 200.
            similarity_flag (bool, optional): Flag to determine whether to use similarity for sampling. Defaults to False.
            dic (bool, optional): Flag to indicate if logs are in dictionary format. Defaults to False.

        Returns:
            list: A list of sampled log strings.
        """
        if dic:
            logs = get_logs_from_group(logs)

        if max_logs is not None and len(logs) > max_logs:
            logs = random.sample(logs, max_logs)

        if len(logs) < sample_size:
            sample_size = len(logs)
        sample_list = []
        selected_logs = []
        if self.similarity_measure == "random":
            sample_list = random.sample(logs, sample_size)
        else:
            for _ in range(sample_size):
                if not sample_list:
                    longest_log_index = max(
                        range(len(logs)), key=lambda x: len(logs[x].split())
                    )
                    selected_logs.append(logs[longest_log_index])
                    sample_list.append(logs[longest_log_index])
                    del logs[longest_log_index]
                else:
                    candidate_distances = self.min_distance(logs, selected_logs)
                    if similarity_flag:
                        best_candidate_index = min(
                            range(len(candidate_distances)),
                            key=lambda x: candidate_distances[x],
                        )
                    else:
                        best_candidate_index = max(
                            range(len(candidate_distances)),
                            key=lambda x: candidate_distances[x],
                        )

                    selected_logs.append(logs[best_candidate_index])
                    sample_list.append(logs[best_candidate_index])
                    logs.remove(logs[best_candidate_index])

        return [log.replace(",", "") for log in sample_list]

    def __check_long_logs(self, log_list, is_dict=False):
        """
        Check if the log list contains long logs that match a specific pattern.

        Args:
            log_list (list): A list of log strings or dictionaries containing logs.
            is_dict (bool, optional): Flag to indicate if logs are in dictionary format. Defaults to False.

        Returns:
            bool: True if the log list contains long logs that match the pattern, False otherwise.
        """
        if is_dict:
            log_list = get_logs_from_group(log_list)
        if len(log_list[0].split()) > 100 and verify_one_regex(
            log_list[0], "Warning: we failed to resolve data source name (.*?)$"
        ):
            return True
        return False

    def __template_to_regex(self, template):
        """
        Convert a log template to a regex pattern.

        Args:
            template (str): The log template.

        Returns:
            str: The regex pattern.
        """
        template = template.strip()
        if "chatglm" in self.model:
            template = replace_bracketed_uppercase(
                template.replace("Log Template: ", "").strip()
            )
        while template.startswith("```") and template.endswith("```"):
            template = template[4:-4]
        while template.startswith("`"):
            template = template[1:]
        while template.endswith("`"):
            template = template[:-1]
        while template.endswith("."):
            template = template[:-1]
        while template.endswith("<*"):
            template = template + ">"
        while template.endswith("<"):
            template = template + "*>"
        while template.endswith("\\"):
            template = template[:-1]
        template = re.sub(r"\<\*\d+\*\>", "<*>", template)
        template = re.sub(r"\<\*\d+\*", "<*>", template)
        template = re.sub(r"\<\*\d+", "<*>", template)
        template = re.sub(r"\<\*\d+\*\>", "<*>", template)
        template = (
            template.replace("*<>", "<*>")
            .replace("*<*>", "<*>")
            .replace("<*>*", "<*>")
            .replace("<>*", "<*>")
            .replace("<*|*>", "<*>")
            .replace("<*>>", "<*>")
            .replace("<<*>", "<*>")
            .replace("<*1*>", "<*>")
            .replace("<>", "<*>")
            .replace("<*>.", "<*>")
            .replace(",", "")
        )
        template = re.sub(r"(?!<)\*(?!>)", "<*>", template)
        template = re.sub(r"(?<!<)\*>", "<*>", template)
        escaped = re.escape(template)
        regex_pattern = re.sub(r"<\\\*>", r"(.*?)", escaped)
        regex_pattern = re.sub(r"(\(\.\*\?\))+", r"(.*?)", regex_pattern)
        regex_pattern = regex_pattern.replace("\ ", " ")
        regex_pattern = re.sub(
            "(\(\.\*\?\) ){10,}",
            "(.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?)",
            regex_pattern,
            0,
        )
        regex_pattern = check_and_truncate_regex(regex_pattern)
        return regex_pattern

    def __generalize_regex(self, target_string, regex_pattern):
        """
        Generalize a regex pattern to match a target string.

        Args:
            target_string (str): The target string.
            regex_pattern (str): The regex pattern.

        Returns:
            str: The generalized regex pattern.
        """
        try:
            option_patterns = re.findall(r"\(\?\:(.*?)\)", regex_pattern)
            for option_pattern in option_patterns:
                options = option_pattern.split("|")
                for option in options:
                    modified_pattern = regex_pattern.replace(
                        f"(?:{option_pattern})", option
                    )
                    if re.match(modified_pattern, target_string):
                        return modified_pattern
        except re.error:
            return regex_pattern
        return regex_pattern

    def __correct_single_template(self, template, user_strings=None):
        """
        Correct a single log template by replacing certain patterns with generic placeholders.

        Args:
            template (str): The log template to correct.
            user_strings (list, optional): A list of user-defined strings to replace. Defaults to None.

        Returns:
            str: The corrected log template.
        """
        path_delimiters = {
            r"\s",
            r"\,",
            r"\!",
            r"\;",
            r"\:",
            r"\=",
            r"\|",
            r"\"",
            r"\'",
            r"\[",
            r"\]",
            r"\(",
            r"\)",
            r"\{",
            r"\}",
        }
        token_delimiters = path_delimiters.union(
            {
                r"\.",
                r"\-",
                r"\+",
                r"\@",
                r"\#",
                r"\$",
                r"\%",
                r"\&",
            }
        )
        template = template.replace(
            r"proxy\.((?:[^.]+|\.)*(?:\.-?\d+)+):[0-9]+", "(.*?)"
        )
        template = template.replace(r"\proxy\.([^.]+):(?:-?\d+|443)", "(.*?)")
        template = template.replace(r"(?:.*?:-?\d+)?", "(.*?)")
        template = template.replace(r"(.*?|.*)", "(.*?)")
        template = template.replace(r"(?:\\n|$)", "$")
        template = (
            template.replace(r"(\b)", "")
            .replace(r"\b", "")
            .replace(r"(\n)", "")
            .replace(r"\n", "")
            .replace(r"(?i)", "")
            .replace(r"?i", "")
            .replace(r"(\r)", "")
            .replace(r"\r", "")
        )

        template = template.strip()
        template = re.sub(r"\s+", " ", template)

        tokens = re.split("(" + "|".join(token_delimiters) + ")", template)
        new_tokens = []
        for token in tokens:
            if re.match(r"^\d+$", token):
                token = "(.*?)"

            if re.match(r"^[^\s\/]*<\*>[^\s\/]*$", token):
                if token != "(.*?)/(.*?)":
                    token = "(.*?)"

            new_tokens.append(token)

        template = "".join(new_tokens)

        while True:
            prev = template
            template = re.sub(r"<\*>\.<\*>", "(.*?)", template)
            if prev == template:
                break

        while True:
            prev = template
            template = re.sub(r"<\*><\*>", "(.*?)", template)
            if prev == template:
                break

        while template.endswith("\\"):
            template = template[:-1]
        while " #(.*?)# " in template:
            template = template.replace(" #(.*?)# ", " (.*?) ")

        while " #(.*?) " in template:
            template = template.replace(" #(.*?) ", " (.*?) ")

        while "(.*?):(.*?)" in template:
            template = template.replace("(.*?):(.*?)", "(.*?)")

        while "(.*?)#(.*?)" in template:
            template = template.replace("(.*?)#(.*?)", "(.*?)")

        while "(.*?)/(.*?)" in template:
            template = template.replace("(.*?)/(.*?)", "(.*?)")

        while "(.*?)@(.*?)" in template:
            template = template.replace("(.*?)@(.*?)", "(.*?)")

        while "(.*?).(.*?)" in template:
            template = template.replace("(.*?).(.*?)", "(.*?)")

        while ' "(.*?)" ' in template:
            template = template.replace(' "(.*?)" ', " (.*?) ")

        while " '(.*?)' " in template:
            template = template.replace(" '(.*?)' ", " (.*?) ")

        while "(.*?)(.*?)" in template:
            template = template.replace("(.*?)(.*?)", "(.*?)")

        return template

    def __replace_nth(self, string, old, new, n):
        """
        Replace the nth occurrence of a substring in a string.

        Args:
            string (str): The original string.
            old (str): The substring to be replaced.
            new (str): The new substring to replace the old one.
            n (int): The occurrence number to replace.

        Returns:
            str: The string with the nth occurrence of the substring replaced.
        """
        parts = string.split(old)
        if len(parts) <= n:
            return string
        return old.join(parts[:n]) + new + old.join(parts[n:])

    def __check_and_modify_regex(self, regex_pattern, target_string):
        """
        Check and modify a regex pattern to ensure it matches the target string.

        Args:
            regex_pattern (str): The regex pattern to check and modify.
            target_string (str): The target string to match against the regex pattern.

        Returns:
            str: The modified regex pattern.
        """
        try:
            pattern = re.compile(regex_pattern)
            match = pattern.match(target_string)
        except re.error:
            return regex_pattern

        if not match:
            return regex_pattern
        groups = match.groups()
        if len(groups) != 0 and groups[-1] == "":
            regex_pattern = regex_pattern + "$"
        try:
            pattern = re.compile(regex_pattern)
            match = pattern.match(target_string)
            groups = match.groups()
        except re.error:
            return regex_pattern
        modified_regex = regex_pattern
        for i, group in enumerate(groups, start=1):
            if group is not None and re.fullmatch(r"\*+", group):
                replacement = "\\" + "\\".join(list(group))
                modified_regex = self.__replace_nth(
                    modified_regex, "(.*?)", replacement, i
                )
            if group is not None and group.endswith(" "):
                replacement = "(.*?) "
                modified_regex = self.__replace_nth(
                    modified_regex, "(.*?)", replacement, i
                )
            if group is not None and group.startswith(" "):
                replacement = " (.*?)"
                modified_regex = self.__replace_nth(
                    modified_regex, "(.*?)", replacement, i
                )
        return modified_regex

    def __clean_regex(self, log, regex_pattern):
        """
        Clean and generalize a regex pattern to match a log string.

        Args:
            log (str): The log string to match against the regex pattern.
            regex_pattern (str): The regex pattern to clean and generalize.

        Returns:
            str: The cleaned and generalized regex pattern.
        """
        regex_pattern = (
            regex_pattern.replace(r"\d+\.\d+", r"\d+(\.\d+)?")
            .replace(r"\\d+", r"-?\\d+")
            .replace("a-f", "a-z")
            .replace("A-F", "A-Z")
        )
        regex_pattern = self.__correct_single_template(regex_pattern)
        if log:
            regex_pattern = self.__generalize_regex(log, regex_pattern)
            regex_pattern = self.__check_and_modify_regex(regex_pattern, log)
        return regex_pattern

    def __extract_logs_from_group(self, group_list):
        """
        Extract log contents from a list of group dictionaries.

        Args:
            group_list (list): A list of dictionaries containing log information.

        Returns:
            list: A list of log contents extracted from the group dictionaries.
        """
        logs = []
        for element in group_list:
            logs.append(element["Content"])
        return logs

    def __find_longest_backtick_content(self, text):
        """
        Find the longest content enclosed in backticks in the given text.

        Args:
            text (str): The input text containing backtick-enclosed content.

        Returns:
            str: The longest content found within backticks, or the original text if no backtick content is found.
        """
        is_multiline = "\n" in text
        if is_multiline:
            matches = re.findall(r"`(.*?)`", text)
            if not matches or matches == [""]:
                matches = re.findall(r"`(.*)", text)
            if matches:
                return max(matches, key=len)
        return text

    def __clean_generated_regex(self, log, template):
        """
        Clean and generalize a generated regex pattern to match a log string.

        Args:
            log (str): The log string to match against the regex pattern.
            template (str): The generated regex pattern.

        Returns:
            str: The cleaned and generalized regex pattern.
        """
        template = self.__find_longest_backtick_content(template)
        template = template.strip()
        while template.startswith("`"):
            template = template[1:]
        while template.startswith('"'):
            template = template[1:]
        while template.startswith("^"):
            template = template[1:]
        while template.startswith("\b"):
            template = template[2:]

        while template.endswith("`"):
            template = template[:-1]
        while template.endswith("."):
            template = template[:-1]
        while template.endswith('"'):
            template = template[:-1]
        while template.endswith("$"):
            template = template[:-1]
        while template.endswith("\b"):
            template = template[:-2]
        while template.endswith("finished"):
            template = template[:-8]
        template = template.replace(",", "")
        template = template.replace("\ ", " ")
        return self.__clean_regex(log=log, regex_pattern=template)

    def __check_regex_from_groups(
        self, result_list, group_dict_list, log_regex, new_event=0
    ):
        """
        Check if logs from group dictionaries match a given regex pattern.

        Args:
            result_list (list): A list to store the results.
            group_dict_list (list): A list of dictionaries containing log information.
            log_regex (str): The regex pattern to match against the logs.
            new_event (int, optional): The new event ID. Defaults to 0.

        Returns:
            tuple: A tuple containing the updated result list and the list of wrong logs.
        """
        wrong_logs = []
        for log in group_dict_list:
            if self.do_self_reflection == "True":
                if verify_one_regex(log["Content"], log_regex):
                    self.regex_manager.add_regex_template(log_regex, log["Content"])
                    if new_event == 0:
                        result_list.append((log["Content"], log["EventId"], log_regex))
                    else:
                        result_list.append((log["Content"], new_event, log_regex))
                else:
                    wrong_logs.append(log)
            else:
                if verify_one_regex(log["Content"], log_regex):
                    self.regex_manager.add_regex_template(log_regex, log["Content"])
                result_list.append((log["Content"], log["EventId"], log_regex))
        return result_list, wrong_logs

    def __remove_first_matching_item(self, data, content_to_remove):
        """
        Remove the first item from the list that matches the given content.

        Args:
            data (list): The list of dictionaries containing log information.
            content_to_remove (str): The content to remove from the list.

        Returns:
            list: The updated list with the first matching item removed.
        """
        for i, item in enumerate(data):
            if item["Content"] == content_to_remove:
                del data[i]
                break
        return data
