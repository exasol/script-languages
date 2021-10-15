#!/usr/bin/env python3

from exasol_python_test_framework import udf

class HTTPXMLProcessingTest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    def test_xml_processing(self):
        self.query(udf.fixindent('''
                create or replace java scalar script
                process_users()
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS
                import java.io.IOException;
                import java.io.OutputStream;
                import java.net.InetSocketAddress;
                import java.util.Iterator;
                import java.util.List;
                import java.util.Set;
                import java.util.concurrent.ExecutorService;
                import java.util.concurrent.Executors;
                import java.net.URL;
                import java.net.URLConnection;
                import javax.xml.parsers.DocumentBuilder;
                import javax.xml.parsers.DocumentBuilderFactory;
                import org.w3c.dom.Document;
                import org.w3c.dom.NodeList;
                import org.w3c.dom.Node;
                import org.w3c.dom.Element;
                import com.sun.net.httpserver.Headers;
                import com.sun.net.httpserver.HttpExchange;
                import com.sun.net.httpserver.HttpHandler;
                import com.sun.net.httpserver.HttpServer;
                class PROCESS_USERS {
                    public static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        MyHttpServer hs = new MyHttpServer();
                        int port = hs.start();
                        URL url = new URL("http://localhost:" + port);
                        URLConnection conn = url.openConnection();
                        DocumentBuilder docBuilder = DocumentBuilderFactory.newInstance().newDocumentBuilder();
                        Document doc = docBuilder.parse(conn.getInputStream());
                        NodeList nodes = doc.getDocumentElement().getElementsByTagName("user");
                        for (int i = 0; i < nodes.getLength(); i++) {
                            if (nodes.item(i).getNodeType() != Node.ELEMENT_NODE)
                                continue;
                            Element elem = (Element)nodes.item(i);
                            if (!elem.getAttribute("active").equals("1"))
                                continue;
                            String firstName = elem.getElementsByTagName("first_name").item(0).getChildNodes().item(0).getNodeValue();
                            String lastName = elem.getElementsByTagName("last_name").item(0).getChildNodes().item(0).getNodeValue();
                            ctx.emit(firstName, lastName);
                        }
                        hs.stop();
                    }

                    private static class MyHttpServer {
                        HttpServer server;
                        ExecutorService exService;

                        public int start() throws IOException {
                            server = HttpServer.create(new InetSocketAddress(0), 0);
                            server.createContext("/", new MyHandler());
                            exService = Executors.newCachedThreadPool();
                            server.setExecutor(exService);
                            server.start();
                            return server.getAddress().getPort();
                        }

                        public void stop() throws IOException {
                                    exService.shutdown();
                            server.stop(0);
                        }
                    }

                    private static class MyHandler implements HttpHandler {
                        public void handle(HttpExchange exchange) throws IOException {
                            StringBuilder xml = new StringBuilder();
                            xml.append(\"<?xml version=\'1.0\' encoding=\'UTF-8\'?>\\n    <users>\\n    \");
                            xml.append(\"<user active=\\"1\\">\\n    <first_name>Manuel</first_name>\\n    <last_name>Neuer</last_name>\\n    </user>\\n    \");
                            xml.append(\"<user active=\\"1\\">\\n    <first_name>Joe</first_name>\\n    <last_name>Hart</last_name>\\n    </user>\\n    \");
                            xml.append(\"<user active=\\"0\\">\\n    <first_name>Oliver</first_name>\\n    <last_name>Kahn</last_name>\\n    </user>\\n    \");
                            xml.append(\"</users>\");

                            String requestMethod = exchange.getRequestMethod();
                            if (requestMethod.equalsIgnoreCase("GET")) {
                                Headers responseHeaders = exchange.getResponseHeaders();
                                responseHeaders.set("Content-Type", "text/plain");
                                exchange.sendResponseHeaders(200, xml.length());
                                OutputStream responseBody = exchange.getResponseBody();
                                responseBody.write(xml.toString().getBytes());
                                responseBody.close();
                            }
                        }
                    }
                }
                '''))

        rows = self.query('''
                SELECT process_users()
                FROM DUAL
                ORDER BY lastname
                ''')

        expected = [('Joe', 'Hart'), ('Manuel', 'Neuer')]
        self.assertRowsEqual(expected, rows)


if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

