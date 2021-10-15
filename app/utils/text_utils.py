import re

import calendar


class TextUtils:

    @staticmethod
    def populate_markup(channel_name, final_template):
        pattern = r"\[(.+?)\]"

        if channel_name == "HTML":
            template_obj = {"UL": "<ul>",
                            "/UL": "</ul>",
                            "LI": "<li>",
                            "/LI": "</li>",
                            "NL": "<br/> ",
                            "BOLD": '<span style="font-weight:bold">',
                            "/BOLD": "</span> "
                            }

        else:
            template_obj = {}

        template_tokens = re.findall(pattern, final_template)
        for var in template_tokens:
            try:
                final_template = final_template.replace("[" + var + "]", template_obj[var])
            except:
                print("The following token was not found: {}".format(var))

        return final_template

    @staticmethod
    def bulleted_list(nlg_list):
        nlg_str = ""
        for item in nlg_list:
            nlg_str += "[LI]" + item + "[/LI]"

        nlg_str = "[UL]" + nlg_str + "[/UL]"
        return nlg_str

    @staticmethod
    def newline_list(nlg_list):
        nlg_str = ""
        for item in nlg_list:
            nlg_str += item + "[NL][NL]"
        return nlg_str


    @staticmethod
    def human_format(num):
        try:
            if num < 1000:
                try:
                    return str(int(num))
                except:
                    return '%.0f%s' % (num, '')
            else:
                magnitude = 0
                while abs(num) >= 1000:
                    magnitude += 1
                    num /= 1000.0
                # add more suffixes if you need them

                else:
                    return '%.1f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])
        except:
            return num

    @staticmethod
    def seperated_string(num, dec):
            num_string = '{:,}'.format(int(round(num, dec))).replace(',', ' ')
            return num_string

    @staticmethod
    def round_perc(per_num):
        try:
            return str(round(per_num, 1))
        except:
            return str(per_num)

    @staticmethod
    def remove_whitespace(x):
        try:
            # Remove spaces inside of the string
            x = "".join(x.split())

        except ValueError:
            pass
        return x

    @staticmethod
    def get_prev_month_name(latest_month_name):
        abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
        month_num = abbr_to_num[latest_month_name[:3]]
        if month_num == 1:
            prev_month_num = 12
        else:
            prev_month_num = month_num -1

        prev_month_name = calendar.month_name[prev_month_num][:3]
        return prev_month_name