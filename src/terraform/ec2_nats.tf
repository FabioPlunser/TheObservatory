# EC2-Instanz
resource "aws_instance" "nats_instance" {
  ami                   = "ami-06b21ccaeff8cd686"
  instance_type         = "t2.micro"
  key_name              = "theObservatory"
  vpc_security_group_ids = [aws_security_group.theObservatory_sg.id]


  associate_public_ip_address = true

  user_data = <<-EOF
  #!/bin/bash
  exec > /var/log/user-data.log 2>&1
  set -x

  # Update system and install dependencies
  yum update -y
  yum install -y tar wget

  # Install NATS Server directly (not using Docker)
  cd /tmp
  wget https://github.com/nats-io/nats-server/releases/download/v2.10.7/nats-server-v2.10.7-linux-amd64.tar.gz
  tar -xzf nats-server-v2.10.7-linux-amd64.tar.gz
  cp nats-server-v2.10.7-linux-amd64/nats-server /usr/local/bin/
  
  # Create NATS systemd service
  cat > /etc/systemd/system/nats.service << 'EOL'
  [Unit]
  Description=NATS Server
  After=network.target

  [Service]
  ExecStart=/usr/local/bin/nats-server -js -m 8222
  Restart=always
  User=root

  [Install]
  WantedBy=multi-user.target
  EOL

  # Start NATS service
  systemctl daemon-reload
  systemctl enable nats
  systemctl start nats

  # Set timezone
  timedatectl set-timezone Europe/Vienna
EOF
  iam_instance_profile = "LabInstanceProfile"

  tags = {
    Name = "NatsInstance"
  }

  root_block_device {
    volume_size = 8
    volume_type = "gp2"
    delete_on_termination = true
  }
}

output "nats_instance_public_ip" {
  value = aws_instance.nats_instance.public_ip
}
