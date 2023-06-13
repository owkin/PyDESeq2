#!/usr/bin/env python3
# mypy: ignore-errors

import argparse
import json

import requests


class SlackSendError(Exception):
    pass


def _format_markdown(title, markdown_text, job_status, job_link):
    title = f"*{title}: <{job_link}|{job_status}>*"
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": title}},
        {"type": "section", "text": {"type": "mrkdwn", "text": markdown_text}},
    ]
    return json.dumps({"blocks": blocks})


def send_message(hook_url, title, markdown_text, job_status, job_link):
    response = requests.post(
        hook_url,
        data=_format_markdown(title, markdown_text, job_status, job_link),
        headers={"Content-Type": "application/json"},
    )
    if response.status_code // 100 != 2:
        raise SlackSendError(f"{response.status_code} {response.text}")


def _get_friendly_duration(duration):
    minutes, seconds = divmod(duration, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def main():
    parser = argparse.ArgumentParser()

    class ParseOptInt(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            try:
                setattr(namespace, self.dest, int(values))
            except ValueError:
                setattr(namespace, self.dest, None)

    parser.add_argument("hook")
    parser.add_argument(
        "desc_file", help="Path to a file with the contents of the message"
    )
    parser.add_argument("--title", help="Report title")
    parser.add_argument("--status", help="Reported job status")
    parser.add_argument("--link", help="Link to job")

    args = parser.parse_args()

    try:
        with open(args.desc_file) as fp:
            message = fp.read()
    except FileNotFoundError:
        message = "🔴 Something wrong happened"
    send_message(args.hook, args.title, message, args.status, args.link)


if __name__ == "__main__":
    main()
