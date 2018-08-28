#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
import unicodedata
from udf import useData

def add_uniname(data):
    return [(n, unicodedata.name(unichr(n), 'U+%04X' % n))
            for n in data]

class LuaExpat(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA LUAEXPAT CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA LUAEXPAT')

    def test_luaexpat_basic_leaks(self):
        self.query(udf.fixindent('''
                create or replace lua scalar script
                lxp_open_and_close(times double) returns double as
                require("lxp")
                function run(ctx)
                    local n = ""
                    for i = 1, ctx.times do
                        local p = lxp.new({StartElement = function(parser, name)
                                                              n = name
                                                          end,
                                           EndElement = function(parser, name)
                                                              n = name
                                                        end})
                        p:parse("<hi><this></this></hi>")
                        p:close()
                    end
                    return ctx.times;
                end
                '''))
        rows = self.query('''select lxp_open_and_close(1000000) from dual''')
        self.assertRowEqual((1000000,), rows[0])
    
    def test_luaexpat_crash_with_old_lib_2_0_1(self):
        self.query(udf.fixindent('''
              CREATE OR REPLACE LUA SCALAR SCRIPT
              lua_gensequence(seqlen NUMBER)
              EMITS (seqidx NUMBER) AS
              function run(ctx) 
                if ctx.seqlen == null then
                      return NULL
                end 
                for i = 1,ctx.seqlen do
                    ctx.emit(i)
                end
              end
              '''))
        self.query(udf.fixindent('''
              CREATE OR REPLACE PYTHON SET SCRIPT
              python_tablegen_randxml(quant_emmited_rows INTEGER, max_xml_children INTEGER)
              EMITS (xmlblock VARCHAR(2000000)) AS
              import re
              #import xml.etree.cElementTree as etree
              from lxml import etree
              from random import randint
              import string
              import random
              def run(ctx):
                  while True:
                      for iter_row in range(0, ctx.quant_emmited_rows):
                          root = etree.Element("users")
                          quant_rnd_children = randint(1,ctx.max_xml_children)
                          for iter_xml_child in range(0, quant_rnd_children):
                              curr_active = randint(0,1)
                              curr_user = etree.Element("user", active=str(curr_active))
                              root.append( curr_user )
                              curr_fn = etree.Element("first_name")
                              curr_ln = etree.Element("last_name")
                              curr_user.append(curr_fn)
                              curr_user.append(curr_ln)
                              curr_fn.text = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase) for x in range(randint(4,12)))
                              curr_ln.text = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase) for x in range(randint(4,12)))
                          ctx.emit(etree.tostring(root, encoding='UTF-8'))
                      if not ctx.next():
                          break
              '''))
        rows = self.query('''CREATE OR REPLACE TABLE xml_table_medium AS SELECT python_tablegen_randxml(CAST(10000/128 AS INT), 99) FROM (SELECT lua_gensequence(128)) GROUP BY seqidx''')
        self.query(udf.fixindent('''
                     CREATE OR REPLACE LUA SCALAR SCRIPT
                     lua_getattribute_fromxml_lxp_sax(xmlblock VARCHAR(2000000))
                     EMITS (firstname VARCHAR(99), lastname VARCHAR(99)) AS
                     require("lxp")
                     p = lxp.new(
                         {StartElement = function(p,tag,attr)
                         if tag == "user" and attr.active == "1" then 
                           in_user_tag = true; 
                           current.first_name = ""
                           current.last_name  = "" 
                          end
                         if tag == "first_name" then in_first_name_tag = true; end
                             if tag == "last_name"  then in_last_name_tag  = true; end
                         end,
                         EndElement = function(p, tag)
                         if tag == "user" then in_user_tag = false; 
                             if current.first_name or current.last_name then
                                 users[#users+1] = current
                                 current = {}
                             end
                         end
                         if tag == "first_name" then in_first_name_tag = false; end
                             if tag == "last_name"  then in_last_name_tag  = false; end
                         end,
                         CharacterData = function(p, txt)
                         if in_user_tag then
                             if in_first_name_tag then current.first_name = current.first_name .. txt end
                                 if in_last_name_tag  then current.last_name  = current.last_name  .. txt end
                         end
                     end})

                     function initFields()
                         in_user_tag = false;
                         in_first_name_tag = false;
                         in_last_name_tag = false;
                         current = {}
                         users = {}
                     end

                     function run(ctx)
                         initFields()
                         p:parse(ctx.xmlblock)
                         for i=1,#users do
                             ctx.emit(users[i].first_name, users[i].last_name)
                         end
                     end
   
                     function cleanup()
                         p:parse()
                         p:close()
                     end
                     '''))
        rows = self.query('''SELECT lua_getattribute_fromxml_lxp_sax(xmlblock) FROM xml_table_medium''')
        # old library crash here, we are testing this behaviour for new one, this should work without crash

    def test_lxp_basics(self):
        self.query(udf.fixindent('''
                CREATE or replace lua SCALAR SCRIPT
                lxp_basics(   x varchar(200000)   ) EMITS (var varchar(30), val double) AS
                require("lxp")
                function run(ctx)
                
                  local numCharacterDataCalls = 0
                  local numCommentCalls = 0
                  local numDefaultExpandCalls = 0
                  local numEndCdataSectionCalls = 0
                  local numEndElementCalls = 0
                  local numEndNamespaceDeclCalls = 0
                  local numExternalEntityRefCalls = 0
                  local numNotStandaloneCalls = 0
                  local numNotationDeclCalls = 0
                  local numProcessingInstructionCalls = 0
                  local numStartCdataSectionCalls = 0
                  local numStartElementCalls = 0
                  local numStartNamespaceDeclCalls = 0
                  local numUnparsedEntityDeclCalls = 0
                
                  local p = lxp.new({
                    CharacterData = function(parser, string)
                                      numCharacterDataCalls = numCharacterDataCalls + 1;
                                    end,
                    Comment = function(parser, string)
                                numCommentCalls = numCommentCalls + 1;
                
                              end,
                    DefaultExpand = function(parser, string)
                                      numDefaultExpandCalls = numDefaultExpandCalls + 1;
                                    end,
                    EndCdataSection = function(parser)
                                        numEndCdataSectionCalls = numEndCdataSectionCalls + 1;
                                      end,
                    EndElement = function(parser, elementName)
                                   numEndElementCalls = numEndElementCalls + 1;
                                 end,
                    EndNamespaceDecl = function(parser, namespaceName)
                                         numEndNamespaceDeclCalls = numEndNamespaceDeclCalls + 1;
                                       end,
                    ExternalEntityRef = function(parser, subparser, base, systemId, publicId)
                                          numExternalEntityRefCalls = numExternalEntityRefCalls + 1;
                                        end,
                    NotStandalone = function(parser)
                                      numNotStandaloneCalls = numNotStandaloneCalls + 1;
                                    end,
                    NotationDecl = function(parser, notationName, base, systemId, publicId)
                                     numNotationDeclCalls = numNotationDeclCalls + 1;
                                   end,
                    ProcessingInstruction = function(parser, target, data)
                                              numProcessingInstructionCalls = numProcessingInstructionCalls + 1;
                                            end,
                    StartCdataSection = function(parser)
                                          numStartCdataSectionCalls = numStartCdataSectionCalls + 1;
                                        end,
                    StartElement = function(parser, elementName, attributes)
                                     numStartElementCalls = numStartElementCalls + 1;
                                   end,
                    StartNamespaceDecl = function(parser, namespaceName)
                                           numStartNamespaceDeclCalls = numStartNamespaceDeclCalls + 1;
                                         end,
                    UnparsedEntityDecl = function(parser, enityName, base, systemId, publicId, notationName)
                                           numUnparsedEntityDeclCalls = numUnparsedEntityDeclCalls + 1;
                                         end
                  }, ':');
                
                  p:parse(ctx.x);  p:parse();  p:close();
                
                  ctx.emit('startElementCalls', numStartElementCalls);  ctx.emit('endElementCalls', numEndElementCalls); 
                  ctx.emit('characterDataCalls',  numCharacterDataCalls); ctx.emit('commentCalls',  numCommentCalls);
                  ctx.emit('defaultExpandCalls',  numDefaultExpandCalls);
                  ctx.emit('startCdateSectionCalls',  numStartCdataSectionCalls);
                  ctx.emit('endCdataSectionCalls',  numEndCdataSectionCalls);
                  ctx.emit('startNamespaceDeclCalls',  numStartNamespaceDeclCalls);
                  ctx.emit('endNamespaceDeclCalls',  numEndNamespaceDeclCalls);
                  ctx.emit('externalEntityRefCalls',  numExternalEntityRefCalls);
                  ctx.emit('notStandaloneCalls',  numNotStandaloneCalls);
                  ctx.emit('notationDeclCalls',  numNotationDeclCalls);
                  ctx.emit('processingInstructionCalls',  numProcessingInstructionCalls);
                  ctx.emit('unparsedEntityDeclCalls',  numUnparsedEntityDeclCalls);
                end
                '''))
        rows = self.query(udf.fixindent('''
                select lxp_basics('<?xml version="1.0" standalone="yes"?>
                <!DOCTYPE dummy [<!NOTATION gif SYSTEM "image/gif">
                <!ENTITY logo2 SYSTEM "images/logo.gif" NDATA gif>]>
                <dummy xmlns="http://www.w3.org/2000">
                <?xml-stylesheet type="text/xsl" href="show.xsl"?>
                <hi>    du   <reference/>   <was>      willst?   </was>   
                <!-- schau an, das ist ein Kommentar -->   <![CDATA[Und das ist CDATA]]>   <denn>      du?  
                 </denn>   </hi></dummy>') from dual'''))
        self.assertEqual(5,   rows[0][1])
        self.assertEqual(5,   rows[1][1])
        self.assertEqual(11,  rows[2][1])
        self.assertEqual(1,   rows[3][1])
        self.assertEqual(22,  rows[4][1])
        self.assertEqual(1,   rows[5][1])
        self.assertEqual(1,   rows[6][1])
        self.assertEqual(1,   rows[7][1])
        self.assertEqual(1,   rows[8][1])
        self.assertEqual(0,   rows[9][1])
        self.assertEqual(0,   rows[10][1])
        self.assertEqual(1,   rows[11][1])
        self.assertEqual(1,   rows[12][1])
        self.assertEqual(1,   rows[13][1])
        #
        #
        rows = self.query(udf.fixindent('''
                select lxp_basics('<?xml version="1.0" standalone="no"?>
                <!DOCTYPE s1 PUBLIC "http://dummy.de/dummy.dtd" "dummy.dtd">
                <dummy xmlns="http://www.w3.org/2000">
                </dummy>') from dual'''))
        self.assertEqual(1,  rows[10][1])
        #
        #
        rows = self.query(udf.fixindent('''
                select lxp_basics('<?xml version="1.0" standalone="no"?>
                <!DOCTYPE ttt [<!ENTITY sec1 SYSTEM "sec1.xml">]>
                <ttt>&sec1;</ttt>') from dual'''))
        self.assertEqual(1,  rows[9][1])




    data = add_uniname((
                65,
                255,
                382,
                65279,
                63882,
                66432,
                173746,
                1114111,
                ))
    
    @useData(data)
    def test_luaexpat_unicode(self, unichar, _name):
        self.query(udf.fixindent('''
                create or replace lua scalar script
                unic(xml varchar(200)) returns boolean as
                require("lxp")
                function run(ctx)
                    local res = false
                    local p = lxp.new({StartElement = function(parser, name)
                                           if res == false then
                                              res = name
                                           else
                                              res = false
                                           end
                                       end,
                        EndElement = function(parser, name)
                                           if res == name then
                                               res = true
                                           else
                                               res = false
                                           end
                                     end})
                    p:parse(ctx.xml)
                    p:close()
                    return res;
                end'''))
        rows = self.query('''select unic('<hi>'||unicodechr(%d)||'</hi>') from dual''' % unichar )
        self.assertRowEqual((True,), rows[0])


if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

