# Security Policy

## Supported versions

Tenzor is currently at **v0.1.0 (alpha)**. Only the latest tagged release is supported with security fixes. Older alphas will not receive backports.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

Please **do not** file public GitHub issues for security problems. Instead, email `lee.morton@gmail.com` with:

- A description of the issue and the impact you believe it has.
- Steps to reproduce (proof-of-concept code is appreciated).
- Affected versions / platforms / backends.
- Any suggested mitigation, if you have one.

You should expect an acknowledgement within **7 days**, and a status update at least every **14 days** until the issue is resolved or closed. Once a fix is available, a coordinated disclosure timeline will be agreed before the details are made public.

## Scope

The following are in scope:

- Memory-safety bugs in C++ code (out-of-bounds reads/writes, use-after-free, leaks reachable from the public API).
- Logic bugs in autograd, dispatch, or backend kernels that can lead to incorrect results in a way an attacker could exploit (e.g. via crafted serialized models).
- Deserialization vulnerabilities in the model checkpoint or ONNX import paths.
- Python binding issues that allow escaping intended sandboxing.

The following are generally **out of scope**:

- DoS via legitimate but expensive workloads (large allocations, etc.).
- Issues only reproducible by running attacker-controlled code with full process privileges.
- Bugs in third-party dependencies — please report those upstream.
