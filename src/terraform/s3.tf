resource "aws_s3_bucket" "theobservatory" {
  bucket = "theobservatory"
}

resource "aws_s3_bucket_cors_configuration" "theobservatory" {
  bucket = aws_s3_bucket.theobservatory.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

resource "aws_s3_bucket_public_access_block" "theobservatory" {
  bucket = aws_s3_bucket.theobservatory.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
