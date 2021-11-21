from colorama import Fore, Back, Style


def render_text(tokens, correct, was_updated, masked):
    output = []
    for t, c, u, m in zip(tokens, correct, was_updated, masked):
        # color
        if c:
            if m:
                output.append(Back.GREEN)
            else:
                output.append(Style.RESET_ALL)
        else:
            if u and not m:
                output.append(Back.RED)
            elif u:
                output.append(Back.BLUE)
            else:
                output.append(Style.RESET_ALL)
        # spacing
        output.append(t)
    output.append(Style.RESET_ALL)
    return ' '.join(output)


# def render_text(decoded, correct, was_updated, masked, offsets):
#     for o, c, u, m in reversed(list(zip(offsets, correct, was_updated, masked))):
#         # color
#         style = None
#         if c:
#             if m:
#                 style = Back.GREEN
#             else:
#                 style = Style.RESET_ALL
#         else:
#             if u and not m:
#                 style = Back.RED
#             elif u:
#                 style = Back.BLUE
#             else:
#                 style = Style.RESET_ALL
#         start, end = o
#         # spacing
#         decoded = decoded[:start] + style + decoded[start:]
#
#     decoded += Style.RESET_ALL
#     return decoded
