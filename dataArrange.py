def get_last_chars_until(text, stop_char):
    """
    Gets the last characters of a string until a specified character is reached,
    excluding the stop character itself.

    Args:
        text: The input string.
        stop_char: The character to stop at.

    Returns:
        The substring containing the last characters up to (but not including)
        the stop character, or an empty string if the stop character is not found.
        Returns the original string if the stop character is at the begining of the string.
    """
    try:
        index = text.rindex(stop_char) #find the last occurence of the stop char
        if index == 0:
            return text
        return text[index + 1:]
    except ValueError:
        return ""  # Stop character not found
def dataRearrange1(input_list, augmentation):
    new_list = []
    for i in range(1, augmentation + 1):
        for string in input_list:
            if i == int(get_last_chars_until(string, "-")):
                new_list.append(string)
    return new_list