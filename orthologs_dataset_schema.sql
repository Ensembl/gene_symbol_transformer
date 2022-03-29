-- See the NOTICE file distributed with this work for additional information
-- regarding copyright ownership.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.


DROP DATABASE IF EXISTS orthologs;

CREATE DATABASE IF NOT EXISTS orthologs CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE orthologs;


CREATE TABLE odb10v1_OGs (
  -- OG unique id (not stable between releases)
  id CHAR(128),
  -- level tax_id on which the group was built
  tax_id CHAR(128),
  -- OG name (the most common gene name within the group)
  name VARCHAR(1024),

  PRIMARY KEY (id)
);
