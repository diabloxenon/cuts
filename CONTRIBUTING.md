# The N3N Division Contributor Guidelines

Greetings,

Are you ready to contribute to our project?

We're excited to have you on board! This project is led by the team at the NCRYPTO Opensource Privacy and Security Norms division, affectionately known as [N3N](https://n3n.org). Your interest in contributing is highly appreciated, and we encourage individuals from all backgrounds who share a passion for creating exceptional software.

N3N is driven by the principles of Privacy, Independency, Transparency, and Security. Our mission is to empower everyone to effortlessly uphold their privacy and security for a cleaner digital life. We achieve this by pushing the boundaries of technology and openly sharing our insights with the broader open-source community. Our commitment to transparency aims to make you more independent, safeguarding your rights to privacy and digital security.

If you have collaboration ideas, feel free to reach out to us at [we@n3n.org](mailto:we@n3n.org).

Thank you for joining us on this journey,

The N3N Authors

## Issues

### Security Vulnerability

First and foremost, as a cybersecurity organization, we prioritize the discovery and swift mitigation of vulnerabilities and associated threats. To report security vulnerabilities in our software, please consult our [security policy].

By the way, at Knytx Labs, we're actively seeking full-time, highly skilled hackers to join our team! Explore our current job opportunities [here](https://knytx.com/jobs).

### Feature Requests

Have ideas on how to enhance our projects? Feel free to propose features by opening a GitHub issue. Be sure to provide details about the feature or change and describe any use cases it would enable.

We'll tag feature requests as `enhancement`, and their status will be regularly updated in the issue comments.

### Bugs

When reporting a bug or unexpected behavior in a project, ensure your issue includes detailed steps to reproduce the behavior. Specify the platform you were using, the steps you took, and any error messages encountered.

Reproducible bugs will be tagged as `bug`, and we'll keep you informed about their status through the comments on the issue.

### Wontfix

Issues that won't be addressed will be closed and tagged as `wontfix`. This decision is typically made if an issue is misaligned with the project vision or falls outside the project's scope. We'll provide detailed reasoning in the issue comments.

## Contribution Workflow

### Open Issues

If you're ready to contribute, start by looking at our open issues tagged as [`help wanted`](../../issues?q=is%3Aopen+is%3Aissue+label%3A"help+wanted") or [`good first issue`](../../issues?q=is%3Aopen+is%3Aissue+label%3A"good+first+issue").

You can comment on the issue to let others know you're interested in working on it or to ask questions.

### Making Changes

0. Before contributing, it is highly advised to review our [community guidelines] to know our policies.

1. Fork [the repository].

2. Choose the version for which you want to develop. The code is divided into the following branches:
   a. master: Current stable version.
   b. series/X.Y: There will be a series branch for each major version release (1.0, 2.0). The master branch will be in sync with the latest series branch.
   c. development: Unstable branch where the development for the next minor or major release is happening.
   d. feature/X: These are branches dealing with a specific bug or feature that is not yet ready to be merged.
   e. The master and series branches are stable, and no breaking changes must be submitted or merged. The development and feature branches are unstable and should not be used in production.

3. Start making your changes in your own separate branch. Ensure that there are no build errors by running the project with your changes locally.

4. Write a test that shows that the bug was fixed or that the feature works as expected.

5. Sign your work. Your signature certifies your submission according to the articles of the [Developer Certificate of Origin](/DCO). The sign-off should be in the form:

   ```text
   Signed-off-by: John Doe <john.doe@example.com>
   ```

   We recommend you use your real name and email for signing commits. This helps in showing your contributions to everyone. If you set your user.name and user.email git configs, you can sign your commit automatically with git commit -s.

6. Open a pull request with a name and description of what you did. You can read more about working with pull requests on GitHub [here](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

7. A maintainer will review your pull request and may ask you to make changes if necessary.

## Code Guidelines

## Language-Specific Standards

- **Rust**: Review our standards [here](https://n3n.org/community/guidelines/rust).
- **Python**: Follow [PEP8 conventions](https://www.python.org/dev/peps/pep-0008/).
- **JavaScript & TypeScript**: We use [Prettier](https://prettier.io/) with default settings.

## Licensing

Unless otherwise specified, all N3N open source projects shall comply with the Rust standard licensing model (MIT + Apache 2.0) and are thereby licensed under a dual license, allowing licensees to choose either MIT OR Apache-2.0 at their discretion.

## Contributor Terms

Thank you for your interest in this N3N open source project. By providing a contribution (new or modified code, other input, feedback, or suggestions, etc.), you agree to these Contributor Terms.

You confirm that each of your contributions has been created by you and that you are the copyright owner. You also confirm that you have the right to provide the contribution to us and that you do it under the Rust dual license model (MIT + Apache 2.0) unless specified.

If you want to contribute something that is not your original creation, you may submit it to the N3N community separately from any contribution, including details of its source and of any license or other restriction (such as related patents, trademarks, agreements, etc.).

Please also note that our projects do follow our [community guidelines] to ensure that they are welcoming places for everyone to contribute. By participating in any of the N3N open source projects, you agree to follow our [community guidelines].

[community guidelines]: https://n3n.org/contrib-terms
