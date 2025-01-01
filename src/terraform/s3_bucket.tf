# S3 bucket for uploaded images to be checked
resource "aws_s3_bucket" "the-observatory-faces-to-check" {
  bucket = "the-observatory-faces-to-check"
}

resource "aws_s3_bucket_versioning" "images_versioning" {
  bucket = aws_s3_bucket.the-observatory-faces-to-check.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket for known faces
resource "aws_s3_bucket" "the-observatory-known-faces" {
  bucket = "the-observatory-known-faces"
}

resource "aws_s3_bucket_versioning" "known_faces_versioning" {
  bucket = aws_s3_bucket.the-observatory-known-faces.id
  versioning_configuration {
    status = "Enabled"
  }
}