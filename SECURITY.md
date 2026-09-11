# Security and privacy

Arwanos stores some of the most personal things a person can write. A bug that exposes that data is treated as a security issue, not an ordinary bug.

## Reporting

**Please don't open a public issue** for anything that could expose personal data or compromise a user's machine.

Report it privately through GitHub: **[Report a vulnerability](https://github.com/GMMB1/Transmitted-Ai/security/advisories/new)**.

Include what the issue is, how to reproduce it (using Demo mode and invented data — never real journal content), and what could be exposed.

## What counts

- Journal entries, conversations, or memory facts written somewhere unexpected — logs, caches, temp files, exports
- Personal data shown in Demo mode
- Personal data that could end up in git or any upload
- Any network request that sends personal content off the machine
- The local web interface being reachable from other devices

## Scope

Arwanos is designed to run locally for a single user, and its web interface binds to `127.0.0.1`. Deliberately exposing it to a network is outside its intended use.
