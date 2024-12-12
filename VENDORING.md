# Vendoring Policies

This document outlines recommended vendoring policies for N3N projects.

## Table of Contents

- [Vendoring Policies](#vendoring-policies)
  - [Table of Contents](#table-of-contents)
  - [Vendoring using tags](#vendoring-using-tags)
  - [Semantic Versioning](#semantic-versioning)
  - [Vendoring Cadence](#vendoring-cadence)

## Vendoring using tags

Commit ID-based vendoring provides little/no information about the updates vendored. To fix this, vendors will now require that repositories use annotated tags along with commit IDs to snapshot commits. Annotated tags by themselves are not sufficient since the same tag can be force-updated to reference different commits.

Each tag should:

- Follow Semantic Versioning rules (refer to the section on "Semantic Versioning").
- Have a corresponding entry in the change tracking document (e.g., CHANGELOG.md, GitHub releases file).

Each repo should:

- Have a change tracking document between tags/releases (e.g., CHANGELOG.md, GitHub releases file).

## Semantic Versioning

Annotated version tags should follow [Semantic Versioning](http://semver.org) policies:

"Given a version number MAJOR.MINOR.PATCH, increment the:

1. **MAJOR** version when you make incompatible API changes.
2. **MINOR** version when you add functionality in a backwards-compatible manner.
3. **PATCH** version when you make backwards-compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format."

## Vendoring Cadence

In order to avoid huge vendoring changes, it is recommended to have a regular cadence for vendoring updates, e.g., monthly.
