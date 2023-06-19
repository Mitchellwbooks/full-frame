# full-frame
An Automated Image Quality and Label Tagger for Photography Workflows


## Goals
Photographers maintaining a large catalog of photographs often don't have time to rate all their photos; 
full-frame is designed to take some burden off the tagging and rating of pictures. 


## Definitions
- Tagging
  - A label or categories for a picture, stored in metadata.
  - Eg. A photo of a Tiger may have labels: "Tiger", "Animal"
- Rating
  - A qualitative assessment for a picture, stored in metadata.
  - EG. A photo of a Tiger with subject in focus, good depth of field, and well framed: "5 Stars"

## Project Goals
1. Automated tagging and rating of pictures
2. Editing Platform Agnostic and Non-Destructive by use of common XML/XMP sidecar metadata file format.
   - Capture One
   - Photoshop
   - Etc
3. File System Portable
   - Allow re-installation of tool to any platform without need for maintaining central database.
4. User Informed
   - User's tags and ratings are trained on to inform the system 

## Technical Reading
### [Architecture](docs%2FArchitecture.md)

## Release Targets
-[ ] v 1.0 - Tagging Model with automatic re-training
-[ ] v 2.0 - Quality Model with automatic re-training