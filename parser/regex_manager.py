import re
import string
from datetime import datetime


def verify_regex_match(log, regex_pattern):
    """
    Verifies if the given log matches the provided regex pattern.

    Args:
        log (str): The log string to be matched.
        regex_pattern (str): The regex pattern to match against the log.

    Returns:
        bool: True if the log matches the regex pattern, False otherwise.
    """
    log = log.replace(",", "")
    regex_pattern = regex_pattern.replace(",", "")
    regex_pattern = f"^{regex_pattern}$"
    try:
        if re.search(regex_pattern, log):
            return True
        else:
            return False
    except re.error:
        return False


def is_punctuation_or_space(input_string):
    """
    Checks if the input string consists only of punctuation or spaces, or if the string
    has less than 3 non-punctuation and non-space characters.

    Args:
        input_string (str): The string to be checked.

    Returns:
        bool: True if the string consists only of punctuation or spaces, or has less than
              3 non-punctuation and non-space characters, False otherwise.
    """
    allowed_chars = string.punctuation + " "
    filtered_string = "".join(
        char for char in input_string if char not in allowed_chars
    )
    return (
        all(char in allowed_chars for char in input_string) or len(filtered_string) < 3
    )


class RegexTemplateManager:
    """
    Manages a collection of regex templates, allowing for the addition, retrieval,
    and matching of regex templates. The templates are stored in a list, sorted
    by their word count, and a set to ensure uniqueness.

    Attributes:
        regex_templates (list): A list of tuples containing word count and regex templates.
        regex_template_set (set): A set of unique regex templates.
        total_time (float): The total time spent on matching regex templates.

    Methods:
        add_regex_template(regex_template, log=False):
            does not consist solely of punctuation or spaces.

        add_regex_templates(regex_template_list):

        get_index_by_length(max_length):
            Finds the index of the first regex template with a word count less than
            or equal to max_length.

        get_regex_templates_by_length(max_length):

        find_matched_regex_template(log):
    """

    def __init__(self):
        self.regex_templates = []
        self.regex_template_set = set()
        self.total_time = 0.0

    def add_regex_template(self, regex_template, log=False):
        """
        Adds a regex template to the manager if it is not already present and
        does not consist solely of punctuation or spaces. The template is
        inserted in the list of templates in a position that maintains the
        list sorted by the word count of the templates.

        Args:
            regex_template (str): The regex template to be added.
            log (str, optional): A log string to verify the regex template against. Defaults to False.

        Returns:
            bool: False if the regex template consists only of punctuation or spaces,
                  or if it is already present in the set of templates. Otherwise, None.
        """
        if is_punctuation_or_space(regex_template):
            return False
        if regex_template in self.regex_template_set:
            return
        else:
            if verify_regex_match(log, regex_template):
                self.regex_template_set.add(regex_template)
            else:
                return
        word_count = regex_template.count(" ") + 1
        regex_template_tuple = (word_count, regex_template)

        if not self.regex_templates:
            self.regex_templates.append(regex_template_tuple)
            return

        insert_index = self.get_index_by_length(word_count)

        while (
            insert_index < len(self.regex_templates)
            and self.regex_templates[insert_index][0] >= word_count
        ):
            insert_index += 1

        self.regex_templates.insert(insert_index, regex_template_tuple)

    def add_regex_templates(self, regex_template_list):
        """
        Adds multiple regex templates to the manager.

        Args:
            regex_template_list (list): A list of regex templates to be added.
        """
        for regex_template in regex_template_list:
            self.add_regex_template(regex_template)

    def get_index_by_length(self, max_length):
        """
        Finds the index of the first regex template with a word count less than or equal to max_length.

        Args:
            max_length (int): The maximum word count of the regex templates to be considered.

        Returns:
            int: The index of the first regex template with a word count less than or equal to max_length.
        """
        left_index, right_index = 0, len(self.regex_templates) - 1
        target_index = len(self.regex_templates)

        while left_index <= right_index:
            mid_index = (left_index + right_index) // 2
            if self.regex_templates[mid_index][0] <= max_length:
                target_index = mid_index
                right_index = mid_index - 1
            else:
                left_index = mid_index + 1

        return target_index

    def get_regex_templates_by_length(self, max_length):
        """
        Retrieves regex templates with a word count less than or equal to max_length.

        Args:
            max_length (int): The maximum word count of the regex templates to be retrieved.

        Returns:
            list: A list of regex templates with a word count less than or equal to max_length.
        """
        start_index = self.get_index_by_length(max_length=max_length + 1)
        end_index = self.get_index_by_length(max_length=max_length - 1)
        return self.regex_templates[start_index:end_index]

    def find_matched_regex_template(self, log):
        """
        Finds the first regex template that matches the given log.

        Args:
            log (str): The log string to be matched against the regex templates.

        Returns:
            str: The first matching regex template, or False if no match is found.
        """
        start_time = datetime.now()
        log_word_count = log.count(" ") + 1
        regex_templates_to_match = self.get_regex_templates_by_length(log_word_count)
        for regex_template in regex_templates_to_match:
            try:
                if verify_regex_match(log, regex_template[1]):
                    time_taken = datetime.now() - start_time
                    self.total_time += time_taken.total_seconds()
                    return regex_template[1]
            except re.error:
                pass
        time_taken = datetime.now() - start_time
        self.total_time += time_taken.total_seconds()
        return False
