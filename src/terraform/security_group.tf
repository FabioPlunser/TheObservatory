resource "aws_security_group" "theObservatory_sg" {
  name        = "theObservatory_sg"
  description = "Security group for theObservatory"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH"
  }

  ingress {
    from_port   = 4222
    to_port     = 4222
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "NATS Client Port"
  }

  ingress {
    from_port   = 8222
    to_port     = 8222
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "NATS Monitoring"
  }

  ingress {
    from_port   = 6222
    to_port     = 6222
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "NATS Clustering"
  }

  # Allow all internal traffic between instances in the security group
  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
